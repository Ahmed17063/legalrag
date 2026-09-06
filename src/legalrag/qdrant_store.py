"""Chargement de l'index Parquet dans Qdrant.

Le Parquet reste la source de verite : c'est lui qui porte le texte, les
metadonnees et les vecteurs, et c'est lui qu'on rejoue. Qdrant est un
cache d'interrogation, reconstructible a tout moment sans repasser par
l'API d'embeddings.

Trois cibles possibles, choisies par la configuration :

* **embarque** (defaut) : Qdrant ecrit dans un repertoire local, sans
  serveur ni Docker. Suffisant pour developper et pour un corpus de cette
  taille. Un seul processus a la fois : le repertoire est verrouille.
* **serveur** : `QDRANT_URL=http://localhost:6333`
* **Qdrant Cloud** : `QDRANT_URL=https://xxx.cloud.qdrant.io:6333` et
  `QDRANT_API_KEY`

L'identifiant d'un point derive du `chunk_id` par UUID deterministe :
recharger deux fois le meme index met a jour les memes points au lieu
d'en creer des doublons.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow.dataset as ds
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

# Espace de noms fixe : le meme chunk_id doit toujours donner le meme
# identifiant de point, y compris entre deux machines.
POINT_NAMESPACE = uuid.UUID("6f3c8b1e-51a7-4f6b-9f2d-0c7a1b4e8d33")

# Colonnes portees en payload. `embedding` en est exclu : c'est le vecteur,
# pas une metadonnee.
PAYLOAD_FIELDS = (
    "chunk_id",
    "document",
    "document_id",
    "chunk_index",
    "text",
    "page_start",
    "page_end",
    "pages",
    "timestamp",
    "source_filenames",
    "token_count",
)

# Champs indexes pour permettre le filtrage a la recherche (restreindre a
# un code, a une plage de pages, a une version horodatee).
_KEYWORD_INDEXES = ("document_id", "timestamp")
_INTEGER_INDEXES = ("page_start", "page_end")


class QdrantStoreError(RuntimeError):
    """Echec de chargement dans Qdrant."""


@dataclass(frozen=True, slots=True)
class QdrantTarget:
    """Ou ecrire, et dans quelle collection."""

    collection: str
    url: str | None = None
    api_key: str | None = None
    path: Path | None = None

    @property
    def is_embedded(self) -> bool:
        return not self.url

    def describe(self) -> str:
        if self.url:
            protege = " (avec cle d'API)" if self.api_key else ""
            return f"serveur {self.url}{protege}, collection {self.collection!r}"
        return f"embarque dans {self.path}, collection {self.collection!r}"


def make_client(target: QdrantTarget) -> QdrantClient:
    """Ouvre la connexion correspondant a la cible."""
    if target.url:
        return QdrantClient(url=target.url, api_key=target.api_key, timeout=60)

    if target.path is None:
        raise QdrantStoreError("ni QDRANT_URL ni QDRANT_PATH : cible indeterminee")

    target.path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(target.path))


def read_dimension(index_path: Path) -> int:
    """Dimension des vecteurs, lue dans le schema de l'index.

    Prise a la source plutot que dans la configuration : une collection
    creee avec la mauvaise taille rejetterait tous les points, avec un
    message peu parlant.
    """
    dataset = ds.dataset(index_path, format="parquet")
    field = dataset.schema.field("embedding")
    size = getattr(field.type, "list_size", None)
    if not size or size < 0:
        raise QdrantStoreError(
            "impossible de determiner la dimension des vecteurs depuis "
            f"{index_path} : le champ 'embedding' n'est pas une liste de taille fixe"
        )
    return int(size)


def point_id(chunk_id: str) -> str:
    """Identifiant Qdrant deterministe derive du chunk_id.

    Qdrant n'accepte que des entiers ou des UUID ; `chunk_id` est une
    chaine hexadecimale. Un UUID version 5 conserve le caractere
    deterministe, donc l'idempotence des rechargements.
    """
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


class QdrantIndex:
    """Collection Qdrant alimentee depuis l'index Parquet."""

    def __init__(self, client: QdrantClient, collection: str, dimension: int) -> None:
        self._client = client
        self._collection = collection
        self._dimension = dimension

    # -- collection --------------------------------------------------------

    def ensure_collection(self, recreate: bool = False) -> bool:
        """Cree la collection si besoin. Renvoie True si elle a ete creee."""
        exists = self._client.collection_exists(self._collection)

        if exists and recreate:
            logger.warning("Suppression de la collection %r", self._collection)
            self._client.delete_collection(self._collection)
            exists = False

        if exists:
            self._check_dimension()
            return False

        self._client.create_collection(
            self._collection,
            vectors_config=models.VectorParams(
                size=self._dimension,
                # Les vecteurs Voyage sont normes : cosinus et produit
                # scalaire donneraient le meme classement. Cosinus reste
                # correct si un jour la source ne l'est plus.
                distance=models.Distance.COSINE,
            ),
        )
        logger.info(
            "Collection %r creee (%d dimensions, cosinus)", self._collection, self._dimension
        )
        self._create_payload_indexes()
        return True

    def _check_dimension(self) -> None:
        info = self._client.get_collection(self._collection)
        params = info.config.params.vectors
        taille = getattr(params, "size", None)
        if taille is not None and taille != self._dimension:
            raise QdrantStoreError(
                f"La collection {self._collection!r} attend des vecteurs de {taille} "
                f"dimensions, l'index en fournit {self._dimension}.\n"
                "Melanger deux modeles rendrait les distances incomparables. "
                "Utiliser --recreate, ou une autre collection."
            )

    def _create_payload_indexes(self) -> None:
        """Index de payload, pour filtrer a la recherche."""
        for champ in _KEYWORD_INDEXES:
            self._client.create_payload_index(
                self._collection, champ, field_schema=models.PayloadSchemaType.KEYWORD
            )
        for champ in _INTEGER_INDEXES:
            self._client.create_payload_index(
                self._collection, champ, field_schema=models.PayloadSchemaType.INTEGER
            )
        logger.debug(
            "Index de payload crees : %s",
            ", ".join(_KEYWORD_INDEXES + _INTEGER_INDEXES),
        )

    # -- chargement --------------------------------------------------------

    def upsert(self, index_path: Path, batch_size: int = 128) -> int:
        """Charge l'index Parquet. Renvoie le nombre de points ecrits.

        Le fichier est lu par lots : l'index tient ici en memoire, mais
        rien n'oblige a ce que ce soit toujours le cas.
        """
        total = 0
        for points in self._iter_points(index_path, batch_size):
            self._client.upsert(self._collection, points=points, wait=True)
            total += len(points)
            logger.info("Charges : %d point(s)", total)
        return total

    def _iter_points(self, index_path: Path, batch_size: int) -> Iterator[list[models.PointStruct]]:
        dataset = ds.dataset(index_path, format="parquet")
        colonnes = [*PAYLOAD_FIELDS, "embedding"]

        for batch in dataset.to_batches(columns=colonnes, batch_size=batch_size):
            lignes = batch.to_pylist()
            if not lignes:
                continue
            yield [
                models.PointStruct(
                    id=point_id(ligne["chunk_id"]),
                    vector=ligne["embedding"],
                    payload={champ: ligne[champ] for champ in PAYLOAD_FIELDS},
                )
                for ligne in lignes
            ]

    # -- entretien ---------------------------------------------------------

    def existing_chunk_ids(self) -> set[str]:
        """chunk_id de tous les points presents dans la collection."""
        trouves: set[str] = set()
        offset = None
        while True:
            points, offset = self._client.scroll(
                self._collection,
                limit=1000,
                offset=offset,
                with_payload=["chunk_id"],
                with_vectors=False,
            )
            trouves.update(p.payload["chunk_id"] for p in points if p.payload)
            if offset is None:
                break
        return trouves

    def prune(self, valid_chunk_ids: set[str]) -> int:
        """Supprime les points absents de l'index Parquet.

        Meme raison que l'elagage cote Parquet : un corpus qui change
        laisse derriere lui des points qui ne correspondent plus a rien et
        qui remonteraient a la recherche.
        """
        obsoletes = self.existing_chunk_ids() - valid_chunk_ids
        if not obsoletes:
            return 0

        self._client.delete(
            self._collection,
            points_selector=models.PointIdsList(points=[point_id(c) for c in obsoletes]),
            wait=True,
        )
        logger.info("Elagage Qdrant : %d point(s) obsolete(s) supprime(s)", len(obsoletes))
        return len(obsoletes)

    def count(self) -> int:
        return self._client.count(self._collection, exact=True).count
