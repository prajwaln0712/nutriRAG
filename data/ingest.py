"""Fetch nutrition details for common food items from the USDA FoodData Central API.

For each food item this script:
  1. Loads the USDA API key from the .env file.
  2. Calls the USDA "foods/search" endpoint.
  3. Extracts calories, protein, carbs, fat and fibre from the response.
  4. Builds a human-readable sentence describing the food.
  5. Saves everything to data/processed/food_item_list.json.

It prints progress for every item and handles API failures, including the
case where the API key has run out of its request quota (rate limit).
"""

import os                       # used to read environment variables and build file paths
import sys                      # used to read the food name from the command line
import json                     # used to write/read the JSON results file
import requests                 # used to make HTTP calls to the USDA API
from dotenv import load_dotenv  # used to load variables defined in the .env file


# A small custom exception so callers can tell a network failure apart from a
# food item that simply was not found in the USDA database.
class NetworkError(Exception):
    """Raised when a USDA API request fails at the network level."""



# USDA FoodData Central search endpoint. We append the query and API key as
# request parameters when we call it below.
USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

# The 20 common food items we want nutrition information for.
FOOD_ITEMS = [
    "Orange", "Apple", "Banana", "Chicken breast", "Egg",
    "Rice", "Broccoli", "Almonds", "Salmon", "Milk",
    "Spinach", "Potato", "Carrot", "Bread", "Cheese",
    "Yogurt", "Lentils", "Tomato", "Oats", "Avocado",
]

# Whole/fresh foods where searching for the "raw" version gives the cleanest
# match. Processed items are deliberately excluded and searched by plain name.
WHOLE_FOODS = {
    "Orange", "Apple", "Banana", "Chicken breast", "Egg", "Broccoli",
    "Almonds", "Salmon", "Spinach", "Potato", "Carrot", "Lentils",
    "Tomato", "Avocado",
}

# The USDA response labels each nutrient with a name. We map the USDA nutrient
# names to the short keys we want to keep in our output.
NUTRIENT_MAP = {
    "Energy": "calories",                    # kilocalories (kcal)
    "Protein": "protein",                    # grams
    "Carbohydrate, by difference": "carbs",  # grams
    "Total lipid (fat)": "fat",              # grams
    "Fiber, total dietary": "fibre",         # grams
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def extract_nutrients(food):
    """Pull the nutrients we care about out of a single USDA food record.

    `food` is one entry from the API's "foods" list. Each food has a
    "foodNutrients" list where every nutrient has a name and a value.
    Returns a dictionary keyed by our short names (calories, protein, ...).
    """
    # Start with every value as None so missing nutrients are clearly absent.
    nutrients = {short_name: None for short_name in NUTRIENT_MAP.values()}

    # Walk through every nutrient reported for this food.
    for nutrient in food.get("foodNutrients", []):
        # The nutrient name can appear under different keys depending on the
        # endpoint, so we check both possibilities.
        name = nutrient.get("nutrientName") or nutrient.get("name")

        # Only keep the value if it is one of the nutrients we are interested in.
        # For "Energy" we additionally make sure the unit is kcal (not kJ).
        if name in NUTRIENT_MAP:
            if name == "Energy":
                unit = nutrient.get("unitName", "").lower()
                if unit and unit != "kcal":
                    continue  # skip the kilojoule version of energy
            nutrients[NUTRIENT_MAP[name]] = nutrient.get("value")

    return nutrients


def build_sentence(name, nutrients):
    """Turn the extracted numbers into a readable sentence.

    Example output:
      "Orange has 47 calories, 0.94 g of protein and 11.75 g of carbs.
       It also has 0.12 g of fats and 2.4 g of fibre in them."
    """
    # Format one nutrient into a phrase. A value of 0 becomes "no <nutrient>";
    # a missing value (None) is reported as "N/A".
    def phrase(label, value, unit="g"):
        if value is None:
            return f"N/A {unit} of {label}"
        if value == 0:
            return f"no {label}"
        return f"{value} {unit} of {label}"

    # Calories carry no "g" unit, so format them on their own.
    if nutrients["calories"] is None:
        calories = "N/A calories"
    elif nutrients["calories"] == 0:
        calories = "no calories"
    else:
        calories = f"{nutrients['calories']} calories"

    protein = phrase("protein", nutrients["protein"])
    carbs = phrase("carbs", nutrients["carbs"])
    fat = phrase("fats", nutrients["fat"])
    fibre = phrase("fibre", nutrients["fibre"])

    return (
        f"{name} has {calories}, {protein} "
        f"and {carbs}. It also has {fat} "
        f"and {fibre} in them."
    )


def fetch_food(query, api_key):
    """Call the USDA API for one food item and return its parsed nutrition.

    Returns a result dictionary on success. Raises RuntimeError when the API
    quota is exhausted so the caller can stop early, and returns None for any
    other failure (network error, no results, etc.).
    """
    # For whole/fresh foods, search the raw version to avoid processed or
    # branded matches; processed items keep their plain name.
    if query in WHOLE_FOODS:
        search_query = f"{query} raw"
    else:
        search_query = query

    # Request parameters: the search term, our API key, a restriction to
    # standardised whole-food datasets, and a single best match.
    params = {
        "api_key": api_key,
        "query": search_query,
        "dataType": ["SR Legacy", "Foundation"],
        "pageSize": 1,
    }

    try:
        # Make the HTTP GET request with a timeout so we never hang forever.
        response = requests.get(USDA_SEARCH_URL, params=params, timeout=30)
    except requests.exceptions.RequestException as exc:
        # Network-level problems (DNS, timeout, connection refused, ...). We
        # raise so callers can tell this apart from a "food not found" result.
        raise NetworkError(str(exc))

    # The USDA API returns HTTP 429 (Too Many Requests) when the key's request
    # quota has been used up. We raise so the main loop can stop gracefully.
    if response.status_code == 429:
        raise RuntimeError("USDA API rate limit reached - the key has run out of tokens.")

    # A 403 with an "OVER_RATE_LIMIT" message also signals an exhausted quota.
    if response.status_code == 403 and "rate limit" in response.text.lower():
        raise RuntimeError("USDA API quota exceeded (403) - the key has run out of tokens.")

    # Any other non-success status is treated as a recoverable per-item error.
    if response.status_code != 200:
        print(f"  [ERROR] API returned status {response.status_code} for '{query}'.")
        return None

    # Parse the JSON body. If the body is not valid JSON, treat it as an error.
    try:
        data = response.json()
    except ValueError:
        print(f"  [ERROR] Could not parse JSON response for '{query}'.")
        return None

    # The "foods" list holds the search results; an empty list means no match.
    foods = data.get("foods", [])
    if not foods:
        print(f"  [WARN] No results found for '{query}'.")
        return None

    # Take the first (best) match and extract its nutrients.
    food = foods[0]
    nutrients = extract_nutrients(food)

    # Assemble the per-food result, including the readable description.
    return {
        "query": query,
        "food_name": food.get("description", query),
        "calories": nutrients["calories"],
        "protein": nutrients["protein"],
        "carbs": nutrients["carbs"],
        "fat": nutrients["fat"],
        "fibre": nutrients["fibre"],
        "description": build_sentence(query, nutrients),
    }


def get_output_path():
    """Return the absolute path to data/processed/food_item_list.json."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "processed", "food_item_list.json")


def load_results(output_path):
    """Read the previously saved food results, or return an empty list."""
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (ValueError, OSError):
            return []  # unreadable or corrupt file - start fresh
    return []


def save_results(results, output_path):
    """Write the food results list to the JSON file (creating folders)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def add_food_if_missing(name, results, api_key):
    """Add `name` to `results` unless an entry for it already exists.

    De-duplicates case-insensitively against the 'query' of existing entries.
    For a missing item it fetches from the USDA API and appends the result.
    Returns a (status, entry) tuple:
      ("exists", entry)        already present; nothing fetched
      ("added",  entry)        fetched from USDA and appended to `results`
      ("not_found", None)      USDA had no such food
      ("network_error", None)  could not reach the USDA API
    Propagates RuntimeError when the API quota is exhausted (callers decide).
    """
    lowered = name.lower()

    # Single source of truth for "is this item already saved?"
    existing = next((r for r in results if r["query"].lower() == lowered), None)
    if existing is not None:
        return "exists", existing

    # Not present - look it up from the USDA API.
    try:
        result = fetch_food(name, api_key)
    except NetworkError:
        return "network_error", None

    # The USDA database returned no match for this food.
    if result is None:
        return "not_found", None

    # New, valid item: append it to the results list.
    results.append(result)
    return "added", result


def lookup_food_item(user_input, api_key=None):
    """Look up a single food item the user typed (the fallback path).

    Reuses add_food_if_missing for the check/fetch/append logic. Returns the
    result dict on success, or a short status string otherwise:
      - "Sorry network error"             if the USDA API could not be reached
      - "Did not find any such food item" if the USDA database has no match
    """
    # Load the API key if the caller did not pass one in.
    if api_key is None:
        load_dotenv()
        api_key = os.getenv("USDA_API_KEY")
        if not api_key:
            print("[FATAL] USDA_API_KEY not found. Check your .env file.")
            return None

    # Clean up the typed name; bail out if it is empty.
    name = user_input.strip()
    if not name:
        print("[WARN] No food item entered.")
        return None

    # Read the saved results so we can both de-duplicate and update them.
    output_path = get_output_path()
    results = load_results(output_path)

    # Run the shared check/fetch/append logic.
    try:
        status, entry = add_food_if_missing(name, results, api_key)
    except RuntimeError as exc:
        # The API quota is exhausted.
        print(f"  [FATAL] {exc}")
        return None

    # Already saved - return the stored values.
    if status == "exists":
        print(f"'{name}' is already in food_item_list.json.")
        print(f"  [OK] {entry['description']}")
        return entry

    # The USDA API could not be reached.
    if status == "network_error":
        message = "Sorry network error"
        print(f"  {message}")
        return message

    # The USDA database has no such food.
    if status == "not_found":
        message = "Did not find any such food item"
        print(f"  {message}")
        return message

    # status == "added": persist the file and record the new item.
    print(f"  [OK] {entry['description']}")
    save_results(results, output_path)
    if not any(item.lower() == name.lower() for item in FOOD_ITEMS):
        FOOD_ITEMS.append(name)
    print(f"  Added '{name}' to food_item_list.json.")
    return entry


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def ingest():
    """Add any missing default food items to the JSON file (created if absent)."""
    # Load variables from the .env file into the environment.
    load_dotenv()

    # Read the USDA API key. Stop early if it is missing.
    api_key = os.getenv("USDA_API_KEY")
    if not api_key:
        print("[FATAL] USDA_API_KEY not found. Check your .env file.")
        return

    # Load the existing file if it exists; otherwise start with an empty list.
    output_path = get_output_path()
    results = load_results(output_path)

    # Loop over every default food item, showing progress as we go.
    for index, food_item in enumerate(FOOD_ITEMS, start=1):
        print(f"[{index}/{len(FOOD_ITEMS)}] Processing '{food_item}'...")

        try:
            status, entry = add_food_if_missing(food_item, results, api_key)
        except RuntimeError as exc:
            # The quota is exhausted - report it and stop fetching further items.
            print(f"  [FATAL] {exc}")
            print("  Stopping early; saving whatever has been collected so far.")
            break

        # Report the outcome for this item.
        if status == "exists":
            print("  [SKIP] already in food_item_list.json.")
        elif status == "network_error":
            print("  [ERROR] Sorry network error.")
        elif status == "not_found":
            print("  [WARN] Did not find any such food item.")
        else:  # "added"
            print(f"  [OK] {entry['description']}")

    # Save the merged list back to disk.
    save_results(results, output_path)

    # Final summary for the user.
    print(f"\nDone. Saved {len(results)} food item(s) to {output_path}.")


if __name__ == "__main__":
    # With a food name argument -> look up that single item (fallback path).
    # With no arguments        -> fetch the full default list.
    if len(sys.argv) > 1:
        lookup_food_item(" ".join(sys.argv[1:]))
    else:
        ingest()
