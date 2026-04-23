import argparse
import json
from pathlib import Path
from bs4 import BeautifulSoup


def parse_match_table(html_text: str) -> dict:
    soup = BeautifulSoup(html_text, "html.parser")
    table = soup.find("table", id="match-table")

    if table is None:
        raise ValueError("Could not find <table id='match-table'>")

    tbody = table.find("tbody")
    if tbody is None:
        raise ValueError("Table is missing <tbody>")

    teams = []

    for row in tbody.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 7:
            continue

        name = cells[0].get_text(strip=True)

        raw_scores = [td.get_text(strip=True) for td in cells[1:7]]
        try:
            scores = [int(x) for x in raw_scores]
        except ValueError as exc:
            raise ValueError(f"Failed parsing numeric scores for team '{name}': {raw_scores}") from exc

        teams.append({
            "name": name,
            "sets": scores[:5],
            "total": scores[5],
        })

    if not teams:
        raise ValueError("No team rows were parsed from the table")

    return {"teams": teams}


def main():
    parser = argparse.ArgumentParser(
        description="Parse an HTML match score table into match_details.json format"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to HTML file containing the match table",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to output JSON file",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.out)

    html_text = input_path.read_text(encoding="utf-8")
    result = parse_match_table(html_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved parsed table to {output_path}")


if __name__ == "__main__":
    main()
