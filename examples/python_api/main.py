"""Extract a typed incident, then draft a notice with a file-backed Kedi program."""

from pathlib import Path

import kedi


@kedi.type
class Incident:
    service: str
    symptom: str
    resolved: bool


@kedi.query
def extract_incident(update: str) -> Incident:
    """kedi
    >> Extract only facts established by this operations update: <update>
    A planned fix is not a completed recovery.
    The incident is [incident: Incident].
    = `incident`
    """
    ...


path = Path(__file__)


@kedi.bind(file=path.with_name("notice.kedi"))
def draft_notice(incident: Incident) -> str:
    """Draft a customer-facing notice from the extracted incident."""
    ...


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    incident = extract_incident(
        "Checkout requests are timing out. We plan to roll back the pool-size "
        "change, but recovery has not been confirmed."
    )
    print(incident)
    print(draft_notice(incident))
