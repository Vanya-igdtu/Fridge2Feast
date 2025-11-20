"""
recipes_groq.py
Robust recipe generator for Groq with optional user-supplied pantry and dietary filters.
- Loads GROQ_API_KEY from .env (via load_dotenv)
- Prompts Groq to return Base64-encoded JSON (preferred)
- Falls back to searching for Base64, extracting balanced JSON, or direct JSON
- Validates recipes against user ingredients + pantry + dietary filters
"""

import os
import json
import time
import re
import requests
import base64
import random
from typing import List, Dict, Any, Optional, Set, Tuple
from dotenv import load_dotenv

load_dotenv()

# === CONFIG ===
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
TEMPERATURE = 0.0
MAX_TOKENS = 1024
MAX_ATTEMPTS = 5
INITIAL_BACKOFF = 0.8

# Default pantry (used when user does not supply one)
DEFAULT_PANTRY: Set[str] = {
    "salt", "pepper", "oil", "butter", "water", "sugar", "flour", "milk",
    "vinegar", "garlic", "onion", "ginger", "soy sauce", "tomato paste"
}

# Ingredients sets for dietary rules (extend as needed)
_ANIMAL_PRODUCTS = {
    "egg", "eggs", "cheese", "milk", "butter", "yogurt", "cream", "honey",
    "meat", "chicken", "beef", "pork", "lamb", "fish", "shrimp", "prawn", "crab", "seafood"
}
_MEAT_AND_FISH = {"meat", "chicken", "beef", "pork", "lamb", "fish", "shrimp", "prawn", "crab"}
_GLUTEN_ITEMS = {"flour", "bread", "pasta", "noodle", "wheat", "breadcrumbs", "chapati", "roti", "naan"}

# === HELPERS ===
B64_CHARS_RE = re.compile(r'[A-Za-z0-9+/=\s]{16,}')  # find long base64-like substrings


def try_b64_decode_whole(text: str) -> Optional[str]:
    """Try to decode the entire text as base64; return decoded string or None."""
    try:
        cleaned = "".join(text.split())
        decoded = base64.b64decode(cleaned, validate=True)
        return decoded.decode("utf-8")
    except Exception:
        return None


def try_b64_decode_search(text: str) -> Optional[str]:
    """Find base64-like substrings and try to decode the longest candidate."""
    candidates = B64_CHARS_RE.findall(text)
    candidates = sorted(set(candidates), key=len, reverse=True)
    for cand in candidates:
        cand_clean = "".join(cand.split())
        try:
            decoded = base64.b64decode(cand_clean, validate=True)
            return decoded.decode("utf-8")
        except Exception:
            continue
    return None


def _find_balanced_block(text: str, open_ch: str, close_ch: str) -> Optional[str]:
    """Return the first balanced block starting at open_ch and ending at its matching close_ch."""
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "\\" and not escape:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
        if not in_string:
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        escape = False
    return None


def extract_json_block(text: str) -> Optional[str]:
    """Extract a balanced JSON array or object block, if present."""
    arr = _find_balanced_block(text, "[", "]")
    if arr:
        return arr
    obj = _find_balanced_block(text, "{", "}")
    return obj


def normalize_ingredients(items: List[str]) -> List[str]:
    """Normalize ingredient list: lowercase, strip, synonyms, dedupe preserving order."""
    synonyms = {
        "eggs": "egg",
        "tomatoe": "tomato",
        "mozzarella cheese": "cheese",
        "milk carton": "milk",
        "onions": "onion",
        "garlic cloves": "garlic"
    }
    final: List[str] = []
    seen = set()
    for it in items:
        if not isinstance(it, str):
            continue
        s = it.strip().lower()
        s = synonyms.get(s, s)
        if s and s not in seen:
            seen.add(s)
            final.append(s)
    return final


def _get_banned_set_from_filters(filters: Optional[List[str]]) -> Set[str]:
    """
    Given a list of filters like ['veg','vegan','gluten-free','non-veg'],
    return a set of banned ingredient tokens that must NOT appear in recipes.
    """
    if not filters:
        return set()
    f = {x.strip().lower() for x in filters if isinstance(x, str)}
    banned: Set[str] = set()
    if "vegan" in f:
        # vegan: ban all obvious animal products (eggs, dairy, meat, seafood, honey)
        banned |= _ANIMAL_PRODUCTS
    elif "veg" in f or "vegetarian" in f:
        # vegetarian: ban meat and fish, but allow dairy/eggs
        banned |= _MEAT_AND_FISH
    # gluten-free: ban common gluten-bearing items
    if "gluten-free" in f or "gluten free" in f:
        banned |= _GLUTEN_ITEMS
    # non-veg: doesn't ban items (explicitly allowed), so no action needed
    return banned


def validate_recipe(recipe: Dict[str, Any], allowed: Set[str], banned: Set[str]) -> bool:
    """
    Validate a single recipe object.
    - allowed: items allowed (normalized detected ingredients + pantry)
    - banned: items explicitly disallowed by dietary filters
    """
    required = {"title", "cook_minutes", "servings", "ingredients", "steps"}
    if not isinstance(recipe, dict):
        return False
    if not required.issubset(set(recipe.keys())):
        return False
    if not isinstance(recipe["title"], str) or not recipe["title"].strip():
        return False
    if not isinstance(recipe["cook_minutes"], int) or recipe["cook_minutes"] < 0:
        return False
    if not isinstance(recipe["servings"], int) or recipe["servings"] <= 0:
        return False
    if not isinstance(recipe["ingredients"], list) or len(recipe["ingredients"]) == 0:
        return False
    for ing in recipe["ingredients"]:
        if not isinstance(ing, str):
            return False
        ing2 = ing.strip().lower()
        if ing2 in banned:
            return False
        if ing2 not in allowed:
            return False
    if not isinstance(recipe["steps"], list) or len(recipe["steps"]) == 0:
        return False
    for st in recipe["steps"]:
        if not isinstance(st, str) or not st.strip():
            return False
    return True


def _filters_to_prompt_text(filters: Optional[List[str]]) -> str:
    """Turn filters list into a human-readable prompt clause for the model."""
    if not filters:
        return ""
    f = [x.strip().lower() for x in filters if isinstance(x, str)]
    clauses = []
    if "vegan" in f:
        clauses.append("Only vegan recipes (no eggs, dairy, meat, fish, honey, or other animal products).")
    elif "veg" in f or "vegetarian" in f:
        clauses.append("Only vegetarian recipes (no meat or fish; dairy and eggs allowed).")
    if "gluten-free" in f or "gluten free" in f:
        clauses.append("Only gluten-free recipes (no wheat, flour, bread, pasta, noodles, breadcrumbs).")
    if "non-veg" in f or "nonveg" in f:
        clauses.append("Non-vegetarian recipes are allowed (meat and seafood are OK).")
    return " ".join(clauses)


# === PROMPT (now parameterized with pantry_list and filters_text) ===
def build_prompt(ingredients: List[str], num: int, pantry_list: List[str], filters_text: str) -> List[Dict[str, str]]:
    """
    Build system+user messages for the chat completion.
    pantry_list should be the list of pantry items allowed for this request.
    filters_text is additional instruction text describing dietary filters.
    """
    allowed = ", ".join(ingredients)
    pantry_text = ", ".join(sorted(pantry_list))
    system = (
        "You are a recipe assistant. YOU MUST RETURN ONLY a single Base64-encoded string and NOTHING ELSE. "
        "The Base64 string decodes to a UTF-8 JSON array. Each element must be an object with: "
        "title (string), cook_minutes (int), servings (int), ingredients (array of strings), steps (array of strings). "
        "Allowed pantry items: " + pantry_text + ". "
        "If impossible, return Base64 of []."
        "if you get (veg and non-veg) or (vegan and non-veg)  as filters, generate 1 vegetarian recipe and 1 non-vegetarian recipe."
    )
    # add filters_text to the user instruction to make model obey dietary constraints
    user = f"Ingredients: [{allowed}]. Create up to {num} recipes using ONLY those ingredients + pantry. {filters_text} Return ONLY the Base64 string."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# === Groq client ===
class GroqRecipeGenerator:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment (.env loaded).")

    def _call(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        """
        Send request to Groq and return (assistant_text, response_preview).
        assistant_text: text to parse (may be content or reasoning fallback).
        response_preview: short printable server response preview for diagnostics.
        """
        payload = {
            "model": MODEL,
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        resp = requests.post(f"{BASE_URL}/chat/completions", json=payload, headers=headers, timeout=60)

        # build safe preview of response body for diagnostics
        try:
            body = resp.text
        except Exception:
            body = ""
        preview = (body[:1500].replace("\n", " ") + "...") if len(body) > 1500 else body.replace("\n", " ")

        # Try parse JSON body if possible
        try:
            data = resp.json()
        except Exception:
            # non-JSON body — return empty assistant content and preview for diagnosis
            return "", f"HTTP {resp.status_code} body preview: {preview}"

        # Attempt to extract content; fall back to reasoning if content empty
        try:
            choices = data.get("choices", [])
            if choices and isinstance(choices, list):
                choice0 = choices[0]
                msg = choice0.get("message", {}) if isinstance(choice0, dict) else {}
                content = msg.get("content")
                if content:
                    return str(content), f"HTTP {resp.status_code} JSON preview: {preview}"
                reasoning = msg.get("reasoning")
                if reasoning:
                    return str(reasoning), f"HTTP {resp.status_code} JSON preview: {preview} (used reasoning fallback)"
                return "", f"HTTP {resp.status_code} JSON preview (no content): {preview}"
        except Exception:
            return "", f"HTTP {resp.status_code} JSON preview: {preview}"

        return "", f"HTTP {resp.status_code} JSON preview: {preview}"


    def generate_recipes(
        self,
        raw_ingredients: List[str],
        num_recipes: int = 3,
        user_pantry: Optional[List[str]] = None,
        diet_filters: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Main entrypoint.
        - raw_ingredients: list of strings detected from the fridge
        - num_recipes: desired number of recipes
        - user_pantry: optional list provided by user; if None, DEFAULT_PANTRY is used
        - diet_filters: optional list e.g., ['veg'], ['vegan'], ['gluten-free'], ['non-veg']
        """
        # normalize and prepare allowed sets
        normalized = normalize_ingredients(raw_ingredients)
        if user_pantry:
            pantry_list = [p.strip().lower() for p in user_pantry if isinstance(p, str) and p.strip()]
        else:
            pantry_list = sorted(DEFAULT_PANTRY)

        # build banned set from filters and allowed set
        banned_set = _get_banned_set_from_filters(diet_filters)
        allowed_set: Set[str] = set(normalized) | set(pantry_list)

        # If filters banned some of the detected ingredients, remove them from allowed_set
        # (this will likely lead to no valid recipes, but it's an explicit check)
        allowed_set = allowed_set - banned_set

        # Build human-readable filters text for model prompt
        filters_text = _filters_to_prompt_text(diet_filters)

        messages = build_prompt(normalized, num_recipes, pantry_list, filters_text)

        attempt = 0
        last_err = None
        while attempt < MAX_ATTEMPTS:
            attempt += 1
            try:
                raw, resp_preview = self._call(messages)
                if not raw or raw.strip() == "":
                    # helpful diagnostic when server returned no content
                    raise RuntimeError(f"Empty assistant content. Server response preview: {resp_preview}")

                raw_preview = (raw[:1200].replace("\n", " ") + "...") if len(raw) > 1200 else raw.replace("\n", " ")

                # 1) try whole-string base64 decode
                decoded = try_b64_decode_whole(raw)
                if decoded is not None:
                    parsed = json.loads(decoded)
                else:
                    # 2) try to find base64 substring and decode it
                    decoded = try_b64_decode_search(raw)
                    if decoded is not None:
                        parsed = json.loads(decoded)
                    else:
                        # 3) try to extract a balanced JSON block
                        block = extract_json_block(raw)
                        if block:
                            parsed = json.loads(block)
                        else:
                            # 4) try direct JSON parsing after trimming quotes/backticks
                            candidate = raw.strip().strip('`"\' ')
                            parsed = json.loads(candidate)

                # ensure parsed is a list
                if not isinstance(parsed, list):
                    raise RuntimeError(f"Parsed JSON is not a list (type={type(parsed)}). Raw preview: {raw_preview}")

                # validate each returned recipe and normalize its ingredient names
                valid: List[Dict[str, Any]] = []
                for item in parsed:
                    if validate_recipe(item, allowed_set, banned_set):
                        # normalize ingredient strings inside the recipe
                        item["ingredients"] = [i.strip().lower() for i in item["ingredients"]]
                        valid.append(item)

                # return up to requested number of valid recipes (may be empty if model returns [])
                return valid[:num_recipes]

            except Exception as e:
                last_err = e
                backoff = INITIAL_BACKOFF * (2 ** (attempt - 1)) + random.uniform(0, 0.3)
                print(f"[Attempt {attempt}/{MAX_ATTEMPTS}] parse failed: {repr(e)}")
                # if server preview included, prompt user to check API key/quota/endpoint
                if isinstance(e, RuntimeError) and "Server response preview" in str(e):
                    print(">>> Server response preview included above. Check API key, quota, or endpoint.")
                if attempt < MAX_ATTEMPTS:
                    time.sleep(backoff)
                else:
                    break

        # all attempts exhausted — raise a final error with last error info
        raise RuntimeError(f"Failed to generate recipes after {MAX_ATTEMPTS} attempts. Last error: {last_err}")


# === quick test ===
if __name__ == "__main__":
    # Example: default pantry + vegetarian filter
    # sample = ["onion", "eggs", "tomato", "cheese", "bread"]
    gen = GroqRecipeGenerator()
    # try:
    #     recipes = gen.generate_recipes(sample, num_recipes=3, diet_filters=["non-veg"])
    #     print("Vegetarian recipes (default pantry):")
    #     print(json.dumps(recipes, indent=2, ensure_ascii=False))
    # except Exception as e:diet_filters=["veg", "non-veg"]
    #     print("Error (veg):", str(e))

    # Example: custom pantry + vegan + gluten-free
    sample2 = ["mushroom", "rice", "tomato", "onion", "rice"]  # note: bread is usually gluten -> filter will ban it
    # sample2 = ["mushroom", "jam", "tomato", "onion", "bread"]
    custom_pantry = ["salt", "pepper", "oil", "garlic"]
    try:
        recipes2 = gen.generate_recipes(sample2, num_recipes=3, user_pantry=custom_pantry, )
        # print("\nVegan + Gluten-free recipes (custom pantry):")
        print(json.dumps(recipes2, indent=2, ensure_ascii=False))
    except Exception as e:
        print("Error (vegan+gf):", str(e))
