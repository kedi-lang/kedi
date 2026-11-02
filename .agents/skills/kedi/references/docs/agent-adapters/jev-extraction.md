# Finite Extraction with Jev

Extraction enumerates candidate text locally and asks Jev to select among those
candidates. It is not unconstrained string generation and cannot produce a
missing address or identifier by inference.

````kedi
```
from typing import Annotated
from pydantic import EmailStr

CurrentEmail = Annotated[
    EmailStr | None,
    "The current support email, not the retired address",
]
```

> adapter: pydantic
> model: typesafe/jev-latest

[notice] = Old support: archive@example.org. Current support: help@example.org.
>> From <notice>, the current contact is [email: `CurrentEmail`].
= `email`
````

This live example requires the TypeSafe extra and email validation dependencies.
Native `EmailStr`, supported phone-number schemas, regex patterns, and explicitly
configured extractors can supply finite candidates. Nullable extraction with no
candidates can return `None` without a provider request. Required extraction with
no suitable candidates cannot invent a valid answer.

## Custom Identifiers

For a custom candidate policy, configure the model in Python with
`text_extractors={"ticket_id": RegexExtractor(pattern=r"CASE-\d+")}` from
`kedi_typesafe`, then pass that model to the framework adapter. The dictionary
key selects the output field path expected by the integration. A regex finds
candidates; the field description guides Jev's choice. Test the candidate
recall separately from the selection decision.

Kedi validates the selected typed result. It does not interpret a probability
as a guarantee that an extractor found every relevant candidate.
