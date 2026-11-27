# csv-data-analyst

Agent analyste de données conversationnel, construit avec le [SDK Agent de Mistral](https://docs.mistral.ai/studio/agents/getting-started/quickstart/) (`mistralai-agents`).

L'utilisateur donne un lien Google Drive vers un fichier CSV et pose des questions en langage naturel. L'agent utilise le connecteur `google_drive_mcp` de l'utilisateur (on-behalf-of) pour télécharger le fichier, puis répond avec des chiffres réels grâce à des outils fixes :

- `load_csv_dataset` : charge le CSV téléchargé dans la session et renvoie son résumé ;
- `describe_columns` : décrit chaque colonne (type, nulls, cardinalité, statistiques) ;
- `preview_rows` : renvoie les premières lignes ;
- `filter_rows` : filtre les lignes avec un opérateur fixe (eq, ne, gt, ge, lt, le, contains, isin, is_null, ...) ;
- `aggregate_by_group` : agrégats par groupe (count, sum, mean, median, min, max, std, nunique).

Aucune exécution de code système : `NoSandbox` (pas de shell/fichiers/réseau ; le runtime TypeScript de `run_typescript` n'expose que le catalogue d'outils déclarés), outils propres en `model_access="direct"`, et `PermissionPolicy` qui refuse en dur les outils d'écriture du connecteur Drive (`create_file`, `update_file`, `copy_file`) : le Drive est en lecture seule.

La session (hébergée par Mistral Cloud) conserve l'historique et le jeu de données chargé : les questions de suivi se posent sans se réexpliquer.

## Déploiement (Mistral Cloud, géré)

Déploiement géré Workflows construit depuis ce dépôt GitHub (via le connecteur GitHub) :

- `deployment_name` = `csv-data-analyst`
- `github_url` = ce dépôt, `revision` = branche poussée
- `hardened = true` (requis : le connecteur Google Drive agit pour le compte de l'utilisateur de la session, OBO)

## Utilisation

- Dans Vibe Code Web : lien `https://chat.mistral.ai/managed-agents/csv-data-analyst?...` affiché après le déploiement, ou via Managed Agents dans Mistral Studio.
- Par programme : `agents.Agent.load_from(agent_name="csv-data-analyst")` puis `start_session()` / `session.run(...)`.

## Développement local

```
uv sync
mistral agents info     # inspecte l'agent chargé
mistral agents chat     # chat local (MISTRAL_API_KEY requis)
```
