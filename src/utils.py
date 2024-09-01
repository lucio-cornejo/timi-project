COLUMNA_ID = 'key'

COLUMNA_OBJETIVO = "taxable income amount"

PREDICTORES_NUMERICOS = [
  "age",
  "wage per hour",
  "capital gains",
  "capital losses",
  "dividends from stocks",
  "instance weight",
  "num persons worked for employer",
  "weeks worked in year"
]

PREDICTORES_CATEGORICOS = [
  "class of worker",
  "detailed industry code",
  "detailed occupation code",
  "education",
  "enroll in edu inst last wk",
  "marital stat",
  "major industry code",
  "major occupation code",
  "race",
  "hispanic origin",
  "sex",
  "member of a labor union",
  "reason for unemployment",
  "full or part time employment stat",
  "tax filer stat",
  "region of previous residence",
  "state of previous residence",
  "detailed household and family stat",
  "detailed household summary in household",
  "migration code-change in msa",
  "migration code-change in reg",
  "migration code-move within reg",
  "live in this house 1 year ago",
  "migration prev res in sunbelt",
  "family members under 18",
  "country of birth father",
  "country of birth mother",
  "country of birth self",
  "citizenship",
  "own business or self employed code",
  "fill inc questionnaire for veteran's admin",
  "veterans benefits code",
  "year"
]

VARIABLES_CATEGORICAS = [COLUMNA_OBJETIVO, *PREDICTORES_CATEGORICOS]
