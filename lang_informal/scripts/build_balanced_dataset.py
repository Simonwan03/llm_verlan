#!/usr/bin/env python3
"""Build a balanced 7-group informal French dataset.

The output contains 50 examples for each of these groups:
1. verlan
2. sms
3. spoken
4. verlan_sms
5. verlan_spoken
6. sms_spoken
7. verlan_sms_spoken

This keeps the original CSV schema while expanding the dataset to 350 rows.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

BASE_EXAMPLES = [
    {
        "standard": "Je ne veux pas parler a cette femme ce soir.",
        "verlan": "Je veux pas parler a cette meuf ce soir.",
    },
    {
        "standard": "Je ne vais pas a la fete de mon frere demain.",
        "verlan": "Je vais pas a la teuf de mon reuf demain.",
    },
    {
        "standard": "Je ne trouve pas ce type tres bizarre au fond.",
        "verlan": "Je trouve pas ce mec tres zarbi au fond.",
    },
    {
        "standard": "Je ne peux pas laisser ce policier entrer dans la maison.",
        "verlan": "Je peux pas laisser ce keuf entrer dans la zonmai.",
    },
    {
        "standard": "Je ne supporte plus ce cours tres penible.",
        "verlan": "Je supporte plus ce cours trop relou.",
    },
    {
        "standard": "Je ne veux pas appeler ma mere devant la police.",
        "verlan": "Je veux pas appeler ma reum devant les keufs.",
    },
    {
        "standard": "Je ne comprends pas pourquoi ce type est fou.",
        "verlan": "Je comprends pas pourquoi ce mec est ouf.",
    },
    {
        "standard": "Je ne garde pas mon argent dans cette maison.",
        "verlan": "Je garde pas ma thune dans cette zonmai.",
    },
    {
        "standard": "Je ne peux pas ecouter cette musique avec ce type.",
        "verlan": "Je peux pas ecouter cette zic avec ce mec.",
    },
    {
        "standard": "Je ne vais pas faire ce travail bizarre ce soir.",
        "verlan": "Je vais pas faire ce taf chelou ce soir.",
    },
    {
        "standard": "Je ne veux pas revoir ce policier apres la fete.",
        "verlan": "Je veux pas revoir ce keuf apres la teuf.",
    },
    {
        "standard": "Je ne peux pas suivre cette femme jusque chez elle.",
        "verlan": "Je peux pas suivre cette meuf jusque chez elle.",
    },
    {
        "standard": "Je ne trouve pas cette voiture tres jolie.",
        "verlan": "Je trouve pas cette caisse tres jolie.",
    },
    {
        "standard": "Je ne veux pas vendre ce materiel a ce type.",
        "verlan": "Je veux pas vendre ce matos a ce mec.",
    },
    {
        "standard": "Je ne sais pas pourquoi cette femme est si enervee.",
        "verlan": "Je sais pas pourquoi cette meuf est si venere.",
    },
    {
        "standard": "Je ne veux pas rester dans ce quartier etrange.",
        "verlan": "Je veux pas rester dans ce quartier chelou.",
    },
    {
        "standard": "Je ne peux pas parler a mon frere pendant le travail.",
        "verlan": "Je peux pas parler a mon reuf pendant le taf.",
    },
    {
        "standard": "Je ne comprends pas cette fete bizarre du tout.",
        "verlan": "Je comprends pas cette teuf zarbi du tout.",
    },
    {
        "standard": "Je ne veux pas donner ma voiture a ce policier.",
        "verlan": "Je veux pas donner ma caisse a ce keuf.",
    },
    {
        "standard": "Je ne vais pas laisser ce type fou entrer ici.",
        "verlan": "Je vais pas laisser ce mec ouf entrer ici.",
    },
    {
        "standard": "Tu ne parles pas a cette femme comme ca.",
        "verlan": "Tu parles pas a cette meuf comme ca.",
    },
    {
        "standard": "Tu ne vas pas a la fete sans ton frere.",
        "verlan": "Tu vas pas a la teuf sans ton reuf.",
    },
    {
        "standard": "Tu ne trouves pas ce policier bizarre ?",
        "verlan": "Tu trouves pas ce keuf zarbi ?",
    },
    {
        "standard": "Tu ne gardes pas ton argent dans la maison, j'espere.",
        "verlan": "Tu gardes pas ta thune dans la zonmai, j'espere.",
    },
    {
        "standard": "Tu ne veux pas suivre ce type etrange ce soir.",
        "verlan": "Tu veux pas suivre ce mec chelou ce soir.",
    },
    {
        "standard": "Tu ne peux pas appeler la police maintenant.",
        "verlan": "Tu peux pas appeler les keufs maintenant.",
    },
    {
        "standard": "Tu ne laisses pas ta mere avec ce type fou.",
        "verlan": "Tu laisses pas ta reum avec ce mec ouf.",
    },
    {
        "standard": "Tu ne veux pas quitter ce travail penible ?",
        "verlan": "Tu veux pas quitter ce taf relou ?",
    },
    {
        "standard": "Tu ne comprends pas cette femme bizarre ?",
        "verlan": "Tu comprends pas cette meuf zarbi ?",
    },
    {
        "standard": "Tu ne vas pas vendre cette voiture a ton frere.",
        "verlan": "Tu vas pas vendre cette caisse a ton reuf.",
    },
    {
        "standard": "Est-ce que tu veux parler a cette femme apres la fete ?",
        "verlan": "Est-ce que tu veux parler a cette meuf apres la teuf ?",
    },
    {
        "standard": "Est-ce que tu peux aider ton frere avec ce travail penible ?",
        "verlan": "Est-ce que tu peux aider ton reuf avec ce taf relou ?",
    },
    {
        "standard": "Est-ce que tu trouves ce policier vraiment etrange ?",
        "verlan": "Est-ce que tu trouves ce keuf vraiment chelou ?",
    },
    {
        "standard": "Est-ce que tu vas laisser ton argent dans la maison ?",
        "verlan": "Est-ce que tu vas laisser ta thune dans la zonmai ?",
    },
    {
        "standard": "Est-ce que tu ecoutes cette musique avec ce type ?",
        "verlan": "Est-ce que tu ecoutes cette zic avec ce mec ?",
    },
    {
        "standard": "Est-ce que tu veux revoir ma mere apres le travail ?",
        "verlan": "Est-ce que tu veux revoir ma reum apres le taf ?",
    },
    {
        "standard": "Est-ce que tu peux calmer cette femme tres enervee ?",
        "verlan": "Est-ce que tu peux calmer cette meuf tres venere ?",
    },
    {
        "standard": "Est-ce que tu vas prevenir la police pour cette fete ?",
        "verlan": "Est-ce que tu vas prevenir les keufs pour cette teuf ?",
    },
    {
        "standard": "Est-ce que tu connais ce quartier vraiment bizarre ?",
        "verlan": "Est-ce que tu connais ce quartier vraiment chelou ?",
    },
    {
        "standard": "Est-ce que tu veux acheter ce materiel a ce type ?",
        "verlan": "Est-ce que tu veux acheter ce matos a ce mec ?",
    },
    {
        "standard": "Il y a une femme etrange devant la maison.",
        "verlan": "Il y a une meuf chelou devant la zonmai.",
    },
    {
        "standard": "Il y a un policier pres de la fete ce soir.",
        "verlan": "Il y a un keuf pres de la teuf ce soir.",
    },
    {
        "standard": "Il y a trop d'argent dans la voiture de mon frere.",
        "verlan": "Il y a trop de thune dans la caisse de mon reuf.",
    },
    {
        "standard": "Il y a une musique penible dans ce quartier.",
        "verlan": "Il y a une zic relou dans ce quartier.",
    },
    {
        "standard": "Il y a un type fou devant le travail de ma mere.",
        "verlan": "Il y a un mec ouf devant le taf de ma reum.",
    },
    {
        "standard": "Je ne veux pas que ce policier parle a cette femme.",
        "verlan": "Je veux pas que ce keuf parle a cette meuf.",
    },
    {
        "standard": "Est-ce que tu peux ramener mon frere a la maison ?",
        "verlan": "Est-ce que tu peux ramener mon reuf a la zonmai ?",
    },
    {
        "standard": "Il y a une fete bizarre chez ce type.",
        "verlan": "Il y a une teuf chelou chez ce mec.",
    },
    {
        "standard": "Je ne vais pas preter mon argent a cette femme.",
        "verlan": "Je vais pas preter ma thune a cette meuf.",
    },
    {
        "standard": "Est-ce que tu veux ecouter cette musique avec ton frere ce soir ?",
        "verlan": "Est-ce que tu veux ecouter cette zic avec ton reuf ce soir ?",
    },
]

GROUP_ORDER = [
    "verlan",
    "sms",
    "spoken",
    "verlan_sms",
    "verlan_spoken",
    "sms_spoken",
    "verlan_sms_spoken",
]

GROUP_NOTES = {
    "verlan": "synthetic balanced example; verlan/youth slang lexical substitutions",
    "sms": "synthetic balanced example; sms-style written abbreviations and punctuation drop",
    "spoken": "synthetic balanced example; spoken simplification, negation drop, and oral contractions",
    "verlan_sms": "synthetic balanced example; verlan lexical substitutions + sms-style written abbreviations",
    "verlan_spoken": "synthetic balanced example; verlan lexical substitutions + spoken simplification",
    "sms_spoken": "synthetic balanced example; sms-style written abbreviations + spoken simplification",
    "verlan_sms_spoken": "synthetic balanced example; verlan lexical substitutions + sms-style written abbreviations + spoken simplification",
}


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+([?.!,;:])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def strip_terminal_punctuation(text: str) -> str:
    return re.sub(r"[?.!]+$", "", text.strip())


def drop_negation(text: str) -> str:
    text = re.sub(r"\bne\s+", "", text)
    text = re.sub(r"\bn'", "", text)
    text = re.sub(r"\bNe\s+", "", text)
    text = re.sub(r"\bN'", "", text)
    return text


def apply_replacements(text: str, replacements: list[tuple[str, str]]) -> str:
    updated = text
    for pattern, replacement in replacements:
        updated = re.sub(pattern, replacement, updated)
    return normalize_text(updated)


def to_sms(text: str) -> str:
    replacements = [
        (r"\bpourquoi\b", "pk"),
        (r"\bPourquoi\b", "Pk"),
        (r"\bavec\b", "avc"),
        (r"\bfrere\b", "frr"),
        (r"\bmere\b", "mer"),
        (r"\bfemme\b", "fem"),
        (r"\bmusique\b", "muzik"),
        (r"\bquartier\b", "qartier"),
        (r"\bmaintenant\b", "mnt"),
        (r"\bdemain\b", "dem1"),
    ]
    return strip_terminal_punctuation(apply_replacements(text, replacements))


def to_spoken(text: str) -> str:
    replacements = [
        (r"\bEst-ce que tu\b", "Tu"),
        (r"\bIl y a\b", "Y a"),
        (r"\bJe suis\b", "J'suis"),
        (r"\bJe sais\b", "J'sais"),
        (r"\bJe vais\b", "J'vais"),
        (r"\bJe veux\b", "J'veux"),
        (r"\bJe peux\b", "J'peux"),
        (r"\bJe trouve\b", "J'trouve"),
        (r"\bJe comprends\b", "J'comprends"),
        (r"\bJe supporte\b", "J'supporte"),
        (r"\bJe garde\b", "J'garde"),
        (r"\bJe donne\b", "J'donne"),
        (r"\bJe parle\b", "J'parle"),
        (r"\bJe connais\b", "J'connais"),
        (r"\bJe laisse\b", "J'laisse"),
        (r"\bTu es\b", "T'es"),
        (r"\bTu as\b", "T'as"),
        (r"\bTu veux\b", "T'veux"),
        (r"\bTu peux\b", "T'peux"),
        (r"\bTu vas\b", "T'vas"),
        (r"\bTu trouves\b", "T'trouves"),
        (r"\bTu gardes\b", "T'gardes"),
        (r"\bTu parles\b", "T'parles"),
        (r"\bTu comprends\b", "T'comprends"),
        (r"\bTu connais\b", "T'connais"),
        (r"\bTu ecoutes\b", "T'ecoutes"),
        (r"\bTu laisses\b", "T'laisses"),
    ]
    return apply_replacements(drop_negation(text), replacements)


def transform_sentence(base: dict[str, str], group: str) -> str:
    if group == "verlan":
        return base["verlan"]
    if group == "sms":
        return to_sms(base["standard"])
    if group == "spoken":
        return to_spoken(base["standard"])
    if group == "verlan_sms":
        return to_sms(base["verlan"])
    if group == "verlan_spoken":
        return to_spoken(base["verlan"])
    if group == "sms_spoken":
        return to_spoken(to_sms(base["standard"]))
    if group == "verlan_sms_spoken":
        return to_spoken(to_sms(base["verlan"]))
    raise ValueError(f"Unknown group: {group}")


def build_dataset() -> pd.DataFrame:
    if len(BASE_EXAMPLES) != 50:
        raise ValueError(f"Expected 50 base examples, found {len(BASE_EXAMPLES)}")

    rows: list[dict[str, object]] = []
    row_id = 1

    for group in GROUP_ORDER:
        for base in BASE_EXAMPLES:
            rows.append(
                {
                    "id": row_id,
                    "phenomenon": group,
                    "standard_sentence": base["standard"],
                    "informal_sentence": transform_sentence(base, group),
                    "meaning_same": 1,
                    "notes": GROUP_NOTES[group],
                }
            )
            row_id += 1

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the balanced 350-row dataset.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/paired_informal_french_dataset.csv"),
        help="Path to write the expanded dataset CSV.",
    )
    args = parser.parse_args()

    dataset = build_dataset()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output_csv, index=False)

    print(f"Wrote {len(dataset)} rows to: {args.output_csv}")
    print(dataset["phenomenon"].value_counts().reindex(GROUP_ORDER).to_string())


if __name__ == "__main__":
    main()
