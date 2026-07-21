SYSTEM_SEM = (
    """
    Extract VQA question semantics from text only; do not invent visual details.
    Return ONLY a JSON object {"results":[...]} matching the schema exactly; preserve input order.
    Decision rules:
    1) op priority:
    text(reading cue: written/word/number/say/spell/label/sign/license; brand/logo ONLY if implied visible/printed) >
    count(how many/number of) >
    spat(explicit left/right/top/bottom/foreground/background; where-questions; relations like behind/next to) >
    temp(how long/how old/aged/before/after) >
    cause(why/reason) >
    comp(comparative/superlative) >
    attr(color/material/type/age/duration/name/location asked) >
    exist(only existential presence: "is/are there", "any", "present") >
    ident(who/what/which entity is) >
    act(activity/doing) >
    know(world knowledge beyond simple visual attributes, when no text-reading cue) > other.
    2) If w=yesno AND op=exist => ans=bool, space=bin. Other yes/no use op based on content.
    3) attr consistency: color->ans=color,space=cat; mat->mat,cat; age->age,num; dur->dur,num;
    count->num,num; name->text,cat; loc->loc,cat; type/ident->cat,cat.
    4) txt=true iff op=text. know=true iff op=know.
    5) sp/rel/dx only if explicit in question.
    6) If w=where and op not in {text,know} => op=spat, attr=loc, ans=loc, space=cat.
    """ 
)

SYSTEM_CTL = (
    "Generate 4 control variants for each VQA question. "
    "Return ONLY a JSON object {\"results\":[...]}; preserve input order.\n"
    "Rules:\n"
    "1) deictic_removed: Remove deictic words (this/that/these/those/here/there) "
    "and spatial markers (left/right/top/bottom/foreground/background) while preserving grammar. "
    "Restructure the sentence if needed to remain fluent.\n"
    "2) object_removed: Replace the specific object noun with neutral placeholder 'object'. "
    "Do not replace pronouns or generic nouns already present.\n"
    "3) weaker_object: Replace the specific object noun with a broader hypernym one level up "
    "(dog->animal, car->vehicle, drink->beverage, shirt->clothing, apple->fruit, "
    "knife->utensil, sofa->furniture, bus->vehicle, cat->animal). "
    "If no clear hypernym exists, use 'object'.\n"
    "4) pronominalized: Replace ALL content noun phrases with minimal pronouns. "
    "Grammar-aware rules:\n"
    "   - Attribute ('What color is the dog?') → 'What color is it?'\n"
    "   - Count ('How many cats are there?') → 'How many are there?'; "
    "('How many cats?') → 'How many?'\n"
    "   - Location ('Where is the phone?') → 'Where is it?'\n"
    "   - Action ('What is the man doing?') → 'What is it doing?'\n"
    "   - Yes/no ('Is the dog affectionate?') → 'Is it affectionate?'\n"
    "   - Multi-NP ('between the girl and the giraffe') → 'between them'\n"
    "   Keep ONLY question words, copula, prepositions, predicate/attribute, and pronouns. "
    "Remove all entity and object names. Use 'it'/'they'/'them' — never 'the object'."
)

SYSTEM_CTL_PRONOMINALIZED = (
    "Generate the pronominalized variant for each VQA question. "
    "Return ONLY a JSON object {\"results\":[...]}; preserve input order.\n"
    "Rule:\n"
    "pronominalized: Replace ALL content noun phrases with minimal pronouns. "
    "Grammar-aware rules:\n"
    "   - Attribute ('What color is the dog?') → 'What color is it?'\n"
    "   - Count ('How many cats are there?') → 'How many are there?'; "
    "('How many cats?') → 'How many?'\n"
    "   - Location ('Where is the phone?') → 'Where is it?'\n"
    "   - Action ('What is the man doing?') → 'What is it doing?'\n"
    "   - Yes/no ('Is the dog affectionate?') → 'Is it affectionate?'\n"
    "   - Multi-NP ('between the girl and the giraffe') → 'between them'\n"
    "   Keep ONLY question words, copula, prepositions, predicate/attribute, and pronouns. "
    "Remove all entity and object names. Use 'it'/'they'/'them' — never 'the object'."
)

# ── Schemas ────────────────────────────────────────────────────────────────────

SEM_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "w":     {"type": "string", "enum": ["what","which","who","where","when","why","how","how_many","how_much","how_long","how_old","yesno","other"]},
                    "op":    {"type": "string", "enum": ["exist","ident","attr","count","act","spat","temp","text","know","cause","comp","other"]},
                    "p":     {"type": ["string","null"]},
                    "sub":   {"type": "string"},
                    "obj":   {"type": ["string","null"]},
                    "ent":   {"type": "string", "enum": ["person","animal","vehicle","food","product","place","text","object","other"]},
                    "attr":  {"type": ["string","null"], "enum": ["color","age","count","dur","mat","ident","type","name","loc","other",None]},
                    "ans":   {"type": "string", "enum": ["bool","num","dur","age","color","mat","loc","person","object","cat","text","open"]},
                    "space": {"type": "string", "enum": ["bin","num","cat","open"]},
                    "sp":    {"type": "array", "items": {"type": "string", "enum": ["left","right","top","bottom","front","back","center","foreground","background"]}},
                    "rel":   {"type": "array", "items": {"type": "string", "enum": ["on","in","next","behind","hold","wear","under","above","between","near"]}},
                    "dx":    {"type": "array", "items": {"type": "string", "enum": ["this","that","these","those","here","there"]}},
                    "neg":   {"type": "boolean"},
                    "know":  {"type": "boolean"},
                    "txt":   {"type": "boolean"}, 
                },
                "required": ["w","op","p","sub","obj","ent","attr","ans","space","sp","rel","dx","neg","know","txt","q"]
            }
        }
    },
    "required": ["results"],
    "additionalProperties": False
}

CTL_PRONOMINALIZED_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pronominalized": {"type": "string"}
                },
                "required": ["pronominalized"]
            }
        }
    },
    "required": ["results"],
    "additionalProperties": False
}

CTL_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "deictic_removed":   {"type": "string"},
                    "object_removed":    {"type": "string"},
                    "weaker_object":     {"type": "string"},
                    "pronominalized":     {"type": "string"}
                },
                "required": ["deictic_removed","object_removed","weaker_object","pronominalized"]
            }
        }
    },
    "required": ["results"],
    "additionalProperties": False
}

# ── Translation ────────────────────────────────────────────────────────────────

SYSTEM_TRANSLATE = (
    "You are a professional Korean translator specializing in survey questions. "
    "Translate each English question to natural, concise Korean. "
    "Preserve the meaning exactly. Return ONLY a JSON object {\"results\": [str, ...]} "
    "with one Korean string per input question, in the same order."
)

TRANSLATE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["results"],
    "additionalProperties": False
}
