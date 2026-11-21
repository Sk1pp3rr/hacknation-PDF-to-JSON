"""Pydantic schemas representing the canonical UZP JSON output."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.core.models import DocumentType


# --- 1. Odwołanie (odwolanie.json) ---


CZYNNOSC_OPIS_BY_ID: dict[int, str] = {
    1: "Decyzja o odrzuceniu oferty",
    2: "Ogłoszenie wyboru oferty",
    3: "Decyzja o wykluczeniu wykonawcy",
    4: "Publikacja specyfikacji warunków zamówienia (SWZ)",
    5: "Modyfikacja treści SWZ",
    6: "Odpowiedzi/wyjaśnienia do SWZ",
    7: "Zaniechanie odrzucenia oferty",
    8: "Zaniechanie wykluczenia wykonawcy",
    9: "Decyzja o unieważnieniu postępowania",
    10: "Zaniechanie unieważnienia postępowania",
    11: "Opis przedmiotu zamówienia (OPZ)",
    12: "Projektowane postanowienia umowy",
}


class OdwolanieKIO(BaseModel):
    """Metadata of the KIO appeal and challenged action."""

    data: str | None = Field(default=None, description="Data z nagłówka (YYYY-MM-DD)")
    czynnosc_dzien: str | None = Field(default=None, description="Data czynności dla pierwszego zarzutu (YYYY-MM-DD)")
    czynnosc_powod_id: int = Field(ge=1, le=12, description="ID powodu odwołania (1-12)")
    czynnosc_opis: str = Field(description="Kanoniczny opis zaskarżonej czynności")
    miejscowosc: str | None = Field(default=None, description="Miejscowość z nagłówka")


class PelnomocnikOdwolujacego(BaseModel):
    """Representative details inside an appeal document."""

    imie_nazwisko: str | None = None
    nazwa: str | None = None
    mail: str | None = None


class OdwolujacyItem(BaseModel):
    """Appellant entity."""

    model_config = ConfigDict(populate_by_name=True)

    krs: str | None = None
    nazwa: str
    nip: str | None = None
    regon: str | None = None
    przez_pelnomocnika: int = Field(alias="przez-pelnomocnika", ge=0, le=1)
    pelnomocnik: list[PelnomocnikOdwolujacego] = Field(default_factory=list)

    @field_validator("pelnomocnik", mode="before")
    @classmethod
    def _coerce_pelnomocnik(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, dict):
            return [] if all(v is None for v in value.values()) else [value]
        return value


class ZamawiajacyItem(BaseModel):
    """Contracting authority entity."""

    krs: str | None = None
    nazwa: str
    nip: str | None = None
    regon: str | None = None
    prowadzacy_postepowanie: int | None = Field(default=None, ge=0, le=1)


class ZamowieniePubliczne(BaseModel):
    """Public procurement reference details."""

    numer_bzp: str | None = None
    numer_postepowania: str | None = None
    numer_ted: str | None = None
    tytul: str | None = None


class OdwolanieSchema(BaseModel):
    """Canonical schema for odwolanie.json."""

    OdwolanieKIO: OdwolanieKIO
    Odwolujacy: list[OdwolujacyItem]
    Zamawiajacy: list[ZamawiajacyItem]
    ZamowieniePubliczne: ZamowieniePubliczne


# --- 2. Pełnomocnictwo (pelnomocnictwo.json) ---


class MocodawcaData(BaseModel):
    """Principal entity details."""

    imie_nazwisko: str | None = None
    krs: str | None = None
    nazwa: str
    nip: str | None = None
    typ: Literal["osoba_prawna", "osoba_fizyczna"]


class PodpisujacyItem(BaseModel):
    """Person signing on behalf of the principal."""

    imie_nazwisko: str


class MocodawcaItem(BaseModel):
    """Principal entry with signatories."""

    data: MocodawcaData
    podpisujacy: list[PodpisujacyItem] = Field(default_factory=list)


class PelnomocnikItem(BaseModel):
    """Authorized representative entry."""

    imie_nazwisko: str
    pesel: str | int | None = None


class PelnomocnictwoSchema(BaseModel):
    """Canonical schema for pelnomocnictwo.json."""

    mocodawca: list[MocodawcaItem]
    pelnomocnicy: list[PelnomocnikItem]


# --- 3. Potwierdzenie przelewu (przelew.json) ---


class PrzelewBankowy(BaseModel):
    """Bank transfer details."""

    data_operacji: str | None = None
    kwota: float | None = None
    na_rachunek: str | None = None
    numer_transakcji: str | None = None
    tytul: str | None = None
    z_rachunku: str | None = None


class PrzelewSchema(BaseModel):
    """Canonical schema for przelew.json."""

    przelew_bankowy: PrzelewBankowy


# --- Registry ---


SCHEMA_REGISTRY: dict[DocumentType, type[BaseModel]] = {
    DocumentType.ODWOLANIE: OdwolanieSchema,
    DocumentType.PELNOMOCNICTWO: PelnomocnictwoSchema,
    DocumentType.PRZELEW: PrzelewSchema,
}
