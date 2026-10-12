"""Assign every FFC tree_ordinal source to one of ten analysis domains.

Topics come from the official FFCWS metadata (Metadata Explorer /get_metadata).
A source matches the metadata row whose old_name is unique and equal to it.
In-home sources hv3*/hv4* lost that prefix in the metadata; they match rows of the
same wave in the three in-home surveys whose old_name equals the rest of the name,
choosing by label similarity when two rows qualify. The first listed topic decides
the domain, except that a child-development or education source whose metadata focal
person is a parent and not the child describes the parent and joins parent_education_traits.
"""

import argparse
import difflib
import json
import re

import pandas as pd

DOMAINS = {
    "Demographics": "demographics",
    "Finances": "economy_employment",
    "Employment": "economy_employment",
    "Housing and neighborhood": "housing_neighborhood",
    "Romantic relationships": "family_social_ties",
    "Family and social ties": "family_social_ties",
    "Parenting": "parenting_childcare",
    "Childcare": "parenting_childcare",
    "Cognitive and behavioral development": "child_development_education",
    "Education and school": "child_development_education",
    "Health and health behavior": "health",
    "Attitudes and expectations": "attitudes_expectations",
    "Systems": "systems",
    "Paradata and weights": "none",
}
CHILD_TOPICS = ("Cognitive and behavioral development", "Education and school")
PARENTS = ("fp_mother", "fp_father", "fp_PCG")
IN_HOME_SURVEYS = ("Primary Caregiver", "In Home Survey", "In Home Observations")
IN_HOME_WAVES = {"3": "Year 3", "4": "Year 5"}


def words(label):
    label = re.sub(r"^\s*[a-z0-9_]+[:.]\s*", "", (label or "").lower())
    return re.sub(r"[^a-z0-9 ]", " ", label).split()


def similarity(a, b):
    return difflib.SequenceMatcher(None, words(a), words(b)).ratio()


def domain(row):
    topic = row.topics.split(";")[0]
    if topic in CHILD_TOPICS and any(row[p] == 1 for p in PARENTS) and row.fp_fchild != 1:
        return "parent_education_traits"
    return DOMAINS[topic]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="FFMetadata CSV from metadata.ffcws.princeton.edu/get_metadata")
    parser.add_argument("--background", required=True, help="background.dta, for its variable labels")
    parser.add_argument("--universe", required=True, help="ffc_tree_ordinal.feature_universe.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata, encoding="latin-1", low_memory=False,
                       usecols=["new_name", "old_name", "varlab", "topics", "survey", "wave", "fp_fchild", *PARENTS])
    with pd.io.stata.StataReader(args.background) as reader:
        labels = reader.variable_labels()
    with open(args.universe) as handle:
        sources = [entry["source"] for entry in json.load(handle)["sources"]]

    direct = meta.drop_duplicates("old_name", keep=False).set_index("old_name")
    in_home = meta[meta.survey.isin(IN_HOME_SURVEYS)]
    rows = []
    for source in sources:
        label = labels[source]
        if source in direct.index:
            match, row = "old_name", direct.loc[source]
        elif (name := re.match(r"^hv([34])(.+)$", source)) and not (
                candidates := in_home[(in_home.wave == IN_HOME_WAVES[name[1]]) & (in_home.old_name == name[2])]).empty:
            match = "in_home"
            row = candidates.loc[max(candidates.index, key=lambda i: similarity(label, candidates.varlab[i]))]
        else:
            rows.append({"source": source, "match": "unmatched", "new_name": "", "topics": "",
                         "domain": "none", "label_similarity": "", "ffc_label": label, "metadata_label": ""})
            continue
        rows.append({"source": source, "match": match, "new_name": row.new_name, "topics": row.topics,
                     "domain": domain(row),
                     "label_similarity": round(similarity(label, row.varlab), 3),
                     "ffc_label": label, "metadata_label": row.varlab})

    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    print(table.match.value_counts().to_string())
    print(table.domain.value_counts().to_string())


if __name__ == "__main__":
    main()
