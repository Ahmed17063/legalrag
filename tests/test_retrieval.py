"""Tests de la recherche : decoupage par article, filtrage, reranking."""

from __future__ import annotations

import pytest

from legalrag.retrieval import Passage, Retriever, SearchResult, split_articles


# ---------------------------------------------------------------------------
# Decoupage par article
# ---------------------------------------------------------------------------


def test_decoupage_simple():
    texte = (
        "Article 322- La cour delibere sur la culpabilite de l'accuse et se prononce sur le fait principal. "
        "Article 323- En cas de declaration de culpabilite, la cour delibere sans desemparer sur la peine."
    )
    passages = split_articles(texte)

    assert [num for num, _ in passages] == ["322", "323"]
    assert passages[0][1].startswith("Article 322-")
    assert "culpabilite" in passages[0][1]


def test_preambule_avant_le_premier_article():
    """Ce qui precede le premier en-tete est conserve, sans numero."""
    texte = (
        "suite du paragraphe precedent, coupe en haut de page, qui doit rester dans le passage. "
        "Article 42- Le contenu de l'article quarante-deux, suffisamment long pour ne pas etre fusionne."
    )
    passages = split_articles(texte)

    assert passages[0][0] is None
    assert "suite du paragraphe" in passages[0][1]
    assert passages[1][0] == "42"


@pytest.mark.parametrize(
    "separateur",
    ["-", ".-", ".", ":", " -", "–"],
)
def test_separateurs_rencontres_dans_le_corpus(separateur):
    """L'OCR produit des separateurs varies apres le numero."""
    texte = (
        f"Article 151{separateur} La subvention a l'entretien de l'epouse enceinte n'est pas dechue. "
        f"Article 152{separateur} Le pere est tenu de subvenir a l'entretien de son enfant mineur."
    )
    assert [num for num, _ in split_articles(texte)] == ["151", "152"]


def test_article_majuscules():
    texte = (
        "ARTICLE 151.- La subvention a l'entretien de l'epouse enceinte n'est pas dechue par son indocilite. "
        "ARTICLE 152.- Le pere est tenu de subvenir a l'entretien de son enfant sauf ressources propres."
    )
    assert [num for num, _ in split_articles(texte)] == ["151", "152"]


def test_numero_compose():
    texte = (
        "Article 706-73- La procedure applicable a la criminalite organisee est definie au present titre. "
        "Article 706-74- Les dispositions du present titre sont egalement applicables aux delits connexes."
    )
    assert [num for num, _ in split_articles(texte)] == ["706-73", "706-74"]


# -- le point delicat : renvois contre en-tetes -----------------------------


def test_renvoi_en_minuscule_ne_coupe_pas():
    """Sur ce corpus les renvois sont en minuscules : signal principal."""
    texte = (
        "Article 5- Le signataire de la convention miniere est designe conformement a "
        "l'article 3 ci-dessus, et sous reserve de l'article 2 de la presente convention."
    )
    passages = split_articles(texte)
    assert len(passages) == 1
    assert passages[0][0] == "5"


@pytest.mark.parametrize(
    "renvoi",
    ["ci-dessus", "ci-dessous", "ci-apres", "precite", "de la presente loi", "du present code"],
)
def test_renvoi_capitalise_ne_coupe_pas(renvoi):
    """Meme capitalise, 'Article N ci-dessus' reste un renvoi."""
    texte = (
        "Article 10- Les conditions sont fixees par les dispositions applicables, "
        f"notamment Article 3- {renvoi} qui demeure en vigueur pour toutes les parties."
    )
    passages = split_articles(texte)
    assert [num for num, _ in passages] == ["10"]


def test_chunk_sans_aucun_article_est_rendu_entier():
    """Rien ne doit etre perdu quand le decoupage ne s'applique pas."""
    texte = "Table des matieres. TITRE PRELIMINAIRE : DE L'ACTION PUBLIQUE, page 12."
    passages = split_articles(texte)
    assert len(passages) == 1
    assert passages[0] == (None, texte)


def test_texte_vide():
    assert split_articles("   \n  ") == []


# -- fragments courts -------------------------------------------------------


def test_fragment_trop_court_rattache_au_precedent():
    """Un en-tete orphelin en fin de chunk n'a aucune valeur seul."""
    texte = (
        "Article 100- Un article de longueur normale, avec assez de contenu pour tenir "
        "seul comme passage citable dans une reponse. Article 101-"
    )
    passages = split_articles(texte)

    assert len(passages) == 1
    assert passages[0][0] == "100"
    assert "Article 101-" in passages[0][1]


def test_aucun_caractere_perdu():
    texte = (
        "Preambule de la page. Article 1- Premier contenu suffisamment long pour rester seul. "
        "Article 2- Second contenu egalement assez long pour former un passage distinct."
    )
    recompose = " ".join(m for _, m in split_articles(texte))
    assert texte.split() == recompose.split()


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


def passage(**kwargs) -> Passage:
    base = dict(
        text="Article 326- Aucune personne acquittee legalement ne peut plus etre reprise.",
        document="Code de Procedure Penale",
        document_id="code-de-procedure-penale",
        page_start=127,
        page_end=128,
        pages=(127, 128),
        timestamp="202311230853",
        chunk_id="abc123",
        article="326",
        vector_score=0.378,
    )
    return Passage(**{**base, **kwargs})


def test_citation_sur_plusieurs_pages():
    assert passage().citation() == "Code de Procedure Penale, art. 326 (p. 127-128)"


def test_citation_sur_une_page():
    assert passage(page_start=40, page_end=40).citation() == (
        "Code de Procedure Penale, art. 326 (p. 40)"
    )


def test_citation_sans_numero_d_article():
    assert passage(article=None).citation() == "Code de Procedure Penale (p. 127-128)"


def test_score_privilegie_le_reranking():
    assert passage().score == pytest.approx(0.378)
    assert passage(rerank_score=0.92).score == pytest.approx(0.92)


# ---------------------------------------------------------------------------
# Recherche de bout en bout, sans reseau
# ---------------------------------------------------------------------------

DIMENSION = 4


class FauxPoint:
    def __init__(self, payload, score):
        self.payload = payload
        self.score = score


class FauxReponse:
    def __init__(self, points):
        self.points = points


class FauxQdrant:
    """Repond comme QdrantClient, et memorise les filtres recus."""

    def __init__(self, points):
        self._points = points
        self.filtres = []
        self.ferme = False

    def query_points(self, collection, query, limit, query_filter=None, with_payload=True):
        self.filtres.append(query_filter)
        return FauxReponse(self._points[:limit])

    def close(self):
        self.ferme = True


class LimiteurMuet:
    """Limiteur de debit qui n'attend jamais, et note ce qu'on lui demande."""

    def __init__(self):
        self.acquis = []

    def acquire(self, tokens):
        self.acquis.append(tokens)


class FauxTransport:
    """Repond comme l'API Voyage pour l'embedding et le reranking."""

    def __init__(self):
        self.appels = []
        self.limiter = LimiteurMuet()

    def post(self, endpoint, payload):
        self.appels.append((endpoint, payload))

        if endpoint == "contextualizedembeddings":
            return {
                "data": [{"index": 0, "data": [{"index": 0, "embedding": [1.0] * DIMENSION}]}],
                "usage": {"total_tokens": 12},
            }

        # Reranker de test : plus un document arrive tard dans la liste,
        # plus il marque. L'ordre renvoye est donc l'inverse exact de
        # l'ordre vectoriel, ce qui prouve que le classement vient bien
        # du reranking.
        docs = payload["documents"]
        classes = sorted(
            ({"index": i, "relevance_score": (i + 1) / len(docs)} for i in range(len(docs))),
            key=lambda x: -x["relevance_score"],
        )
        return {"data": classes[: payload["top_k"]], "usage": {"total_tokens": 30}}


def charge(chunk_id, texte, doc="Code de Procedure Penale"):
    return {
        "chunk_id": chunk_id,
        "text": texte,
        "document": doc,
        "document_id": doc.lower().replace(" ", "-"),
        "page_start": 127,
        "page_end": 128,
        "pages": [127, 128],
        "timestamp": "202311230853",
    }


@pytest.fixture
def retriever(settings, monkeypatch):
    from legalrag import retrieval

    points = [
        FauxPoint(
            charge(
                "c1",
                "Article 322- Contenu du premier article, assez long pour former un passage. "
                "Article 326- Aucune personne acquittee legalement ne peut plus etre reprise.",
            ),
            0.38,
        ),
        FauxPoint(charge("c2", "Article 500- Un autre article, sans rapport avec la question posee."), 0.31),
    ]
    faux_qdrant = FauxQdrant(points)
    monkeypatch.setattr(retrieval, "_make_client", lambda s: faux_qdrant)
    monkeypatch.setattr(retrieval, "VoyageTransport", lambda **kw: FauxTransport())

    r = Retriever(settings)
    r._faux_qdrant = faux_qdrant  # pour les assertions
    return r


def test_recherche_decoupe_les_chunks_en_passages(retriever):
    resultat = retriever.search("Une personne acquittee peut-elle etre rejugee ?", rerank=False)

    assert resultat.chunks_examined == 2
    # Le premier chunk porte deux articles : trois passages au total.
    assert resultat.passages_examined == 3
    assert isinstance(resultat, SearchResult)


def test_input_type_query_pour_la_question(retriever):
    """Une question ne doit pas etre plongee comme un document."""
    retriever.search("question", rerank=False)
    endpoint, payload = retriever._transport.appels[0]

    assert endpoint == "contextualizedembeddings"
    assert payload["input_type"] == "query"


def test_reranking_reordonne_les_passages(retriever):
    sans = retriever.search("question", rerank=False, top_n=3)
    avec = retriever.search("question", rerank=True, top_n=3)

    assert not sans.reranked
    assert avec.reranked
    # Le faux reranker inverse l'ordre : la preuve que le classement vient
    # bien de lui et non du vectoriel.
    assert [p.text for p in avec.passages] == [p.text for p in sans.passages][::-1]


def test_les_scores_de_reranking_sont_conserves(retriever):
    resultat = retriever.search("question", rerank=True, top_n=2)
    for p in resultat.passages:
        assert p.rerank_score is not None
        assert p.score == p.rerank_score


def test_top_n_limite_les_passages(retriever):
    assert len(retriever.search("question", rerank=True, top_n=1).passages) == 1
    assert len(retriever.search("question", rerank=False, top_n=2).passages) == 2


def test_filtre_par_document_transmis_a_qdrant(retriever):
    retriever.search("question", document_id="code-du-statut-personnel", rerank=False)
    filtre = retriever._faux_qdrant.filtres[-1]

    assert filtre is not None
    condition = filtre.must[0]
    assert condition.key == "document_id"
    assert condition.match.value == "code-du-statut-personnel"


def test_sans_filtre_par_defaut(retriever):
    retriever.search("question", rerank=False)
    assert retriever._faux_qdrant.filtres[-1] is None


def test_tokens_cumules(retriever):
    """Embedding puis reranking : les deux appels sont comptes."""
    assert retriever.search("question", rerank=False).tokens_billed == 12
    assert retriever.search("question", rerank=True).tokens_billed == 42


def test_metadonnees_conservees_jusqu_au_passage(retriever):
    resultat = retriever.search("question", rerank=True, top_n=3)
    for p in resultat.passages:
        assert p.document == "Code de Procedure Penale"
        assert p.pages == (127, 128)
        assert p.timestamp == "202311230853"
        assert p.chunk_id in {"c1", "c2"}


def test_fermeture_du_client(retriever):
    retriever.close()
    assert retriever._faux_qdrant.ferme


def test_gestionnaire_de_contexte(settings, monkeypatch):
    from legalrag import retrieval

    faux = FauxQdrant([])
    monkeypatch.setattr(retrieval, "_make_client", lambda s: faux)
    monkeypatch.setattr(retrieval, "VoyageTransport", lambda **kw: FauxTransport())

    with Retriever(settings):
        pass
    assert faux.ferme


def test_sortie_json_serialisable(retriever):
    import json

    resultat = retriever.search("question", rerank=True, top_n=2)
    texte = json.dumps(resultat.as_dict(), ensure_ascii=False)

    rejoue = json.loads(texte)
    assert rejoue["reranked"] is True
    assert rejoue["passages"][0]["citation"].startswith("Code de Procedure Penale")


# -- budget de reranking ----------------------------------------------------


def test_reranking_borne_par_le_debit_autorise(settings, monkeypatch):
    """Sur un compte plafonne, on ne peut pas soumettre tous les passages.

    150 passages issus de 25 chunks representent ~16 000 tokens, au-dessus
    d'un plafond de 10 000 par minute : la requete serait refusee a chaque
    tentative.
    """
    from dataclasses import replace

    from legalrag import retrieval

    plafonne = replace(settings, max_tokens_per_minute=10_000, max_requests_per_minute=1000)

    long_texte = "Article 1- " + ("mot " * 300)  # ~340 tokens estimes
    points = [FauxPoint(charge(f"c{i}", long_texte), 0.5 - i / 100) for i in range(25)]
    faux_qdrant = FauxQdrant(points)
    transport = FauxTransport()
    monkeypatch.setattr(retrieval, "_make_client", lambda s: faux_qdrant)
    monkeypatch.setattr(retrieval, "VoyageTransport", lambda **kw: transport)

    with Retriever(plafonne) as r:
        r.search("question", rerank=True, top_n=3)

    envoyes = next(p["documents"] for e, p in transport.appels if e == "rerank")
    cout = sum(retrieval.estimate_tokens(d) for d in envoyes)

    assert len(envoyes) < 25, "tous les passages ont ete soumis malgre le plafond"
    assert cout <= 10_000


def test_aucune_troncature_quand_le_budget_suffit(settings, monkeypatch):
    from legalrag import retrieval

    points = [FauxPoint(charge("c1", "Article 1- Un contenu court mais suffisant pour tenir."), 0.5)]
    monkeypatch.setattr(retrieval, "_make_client", lambda s: FauxQdrant(points))
    transport = FauxTransport()
    monkeypatch.setattr(retrieval, "VoyageTransport", lambda **kw: transport)

    with Retriever(settings) as r:
        r.search("question", rerank=True)

    envoyes = next(p["documents"] for e, p in transport.appels if e == "rerank")
    assert len(envoyes) == 1


def test_estimation_de_tokens():
    assert retrieval_estimate("") == 1
    assert retrieval_estimate("a" * 350) == 100


def retrieval_estimate(texte):
    from legalrag.retrieval import estimate_tokens

    return estimate_tokens(texte)


# ---------------------------------------------------------------------------
# Reformulation de la requete
# ---------------------------------------------------------------------------


def _retriever_avec(settings, monkeypatch, reformulateur):
    """Un Retriever instrumente, dont on peut lire les appels a Voyage."""
    from legalrag import retrieval

    points = [
        FauxPoint(
            charge(
                "c1",
                "Article 326- Aucune personne acquittee legalement ne peut plus etre reprise.",
            ),
            0.38,
        )
    ]
    transport = FauxTransport()
    monkeypatch.setattr(retrieval, "_make_client", lambda s: FauxQdrant(points))
    monkeypatch.setattr(retrieval, "VoyageTransport", lambda **kw: transport)

    return Retriever(settings, reformulateur), transport


def _requete_embarquee(transport) -> str:
    """Le texte reellement envoye au modele d'embedding."""
    for endpoint, payload in transport.appels:
        if endpoint == "contextualizedembeddings":
            return payload["inputs"][0][0]
    raise AssertionError("aucun appel d'embedding")


def _requete_reclassee(transport) -> str:
    for endpoint, payload in transport.appels:
        if endpoint == "rerank":
            return payload["query"]
    raise AssertionError("aucun appel de reranking")


QUESTION = "Peut-on rejuger un acquitte ?"
REECRITE = "autorite de la chose jugee, non bis in idem, personne acquittee rejugee"


def test_le_vectoriel_part_de_la_reformulation(settings, monkeypatch):
    """C'est du vocabulaire que compare un modele d'embedding."""
    r, transport = _retriever_avec(settings, monkeypatch, lambda q: REECRITE)
    resultat = r.search(QUESTION, rerank=False)

    assert _requete_embarquee(transport) == REECRITE
    # La question posee reste celle de l'utilisateur.
    assert resultat.query == QUESTION
    assert resultat.query_reformulee == REECRITE


def test_le_reranking_garde_la_question_d_origine(settings, monkeypatch):
    """Le reranking juge la pertinence : il doit rester sur l'intention."""
    r, transport = _retriever_avec(settings, monkeypatch, lambda q: REECRITE)
    r.search(QUESTION, rerank=True)

    assert _requete_embarquee(transport) == REECRITE
    assert _requete_reclassee(transport) == QUESTION


def test_sans_reformulateur_rien_ne_change(settings, monkeypatch):
    r, transport = _retriever_avec(settings, monkeypatch, None)
    resultat = r.search(QUESTION, rerank=False)

    assert _requete_embarquee(transport) == QUESTION
    assert resultat.query_reformulee is None


def test_reformulation_desactivable_par_appel(settings, monkeypatch):
    """De quoi comparer les deux comportements sur une meme question."""
    r, transport = _retriever_avec(settings, monkeypatch, lambda q: REECRITE)
    resultat = r.search(QUESTION, rerank=False, reformuler=False)

    assert _requete_embarquee(transport) == QUESTION
    assert resultat.query_reformulee is None


def test_reformulation_desactivable_par_configuration(settings, monkeypatch):
    from dataclasses import replace

    sans = replace(settings, reformulation_enabled=False)
    r, transport = _retriever_avec(sans, monkeypatch, lambda q: REECRITE)
    resultat = r.search(QUESTION, rerank=False)

    assert _requete_embarquee(transport) == QUESTION
    assert resultat.query_reformulee is None


def test_reformulation_inchangee_n_est_pas_signalee(settings, monkeypatch):
    """Rien a montrer a l'utilisateur si la requete n'a pas bouge."""
    r, transport = _retriever_avec(settings, monkeypatch, lambda q: q)
    resultat = r.search(QUESTION, rerank=False)

    assert _requete_embarquee(transport) == QUESTION
    assert resultat.query_reformulee is None
