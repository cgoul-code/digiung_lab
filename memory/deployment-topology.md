---
name: deployment-topology
description: How the digiung_lab query app is deployed (separate client/API origins) and the auth implication
metadata:
  type: project
---

digiung_lab is deployed as **two separate Azure services on different origins**:
- **Client**: Azure Static Web App (`witty-beach-06ba13e03…azurestaticapps.net`), built from
  `digiung_lab-query-client` (Vite/React, single `src/App.jsx`). Deploy workflow:
  `.github/workflows/azure-static-web-apps-witty-beach-*.yml`.
- **API**: Azure App Service (Web App `Lab-Document-Query`,
  `lab-document-query-…swedencentral-01.azurewebsites.net`), the Quart server `query_server.py`.
  Deploy workflow: `.github/workflows/main_lab-document-query.yml`.

The client's API base URL is hardcoded in `App.jsx` (`webserverEndPoint`): `http://localhost:80`
in dev, the azurewebsites.net URL in prod. The API is API-only (`cors(app, allow_origin="*")`)
and does **not** serve the client.

**Auth implication:** because they're separate origins with `CORS *` and no credentials,
App Service "Easy Auth" on the API does **not** receive the user's identity from the browser's
cross-origin fetches. Identity must come from either (a) Static Web Apps built-in auth
(`/.auth/me`, same origin as the client — good for per-browser namespacing only), or
(b) linking the App Service as the SWA's backend so SWA injects the `x-ms-client-principal`
header into API calls (enables true cross-device, server-side per-user data).
`query_server.py` `_current_user()` already decodes both `X-MS-CLIENT-PRINCIPAL-NAME` and the
base64 `X-MS-CLIENT-PRINCIPAL` claims, so it's ready for the linked-backend path.
Per-user conversation log lives at `INDEX_STORAGE/_user_history/<user>.json`. See [[storage-architecture]].
