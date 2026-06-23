# Memory index

- [Storage architecture](storage-architecture.md) — digiung_lab indexes live in the `labdocumentquery` Azure Files share, not the `chatbot` Blob container (which is the llama_chatbot app's).
- [Deployment topology](deployment-topology.md) — client (Static Web App) and API (App Service) are separate origins; auth/identity implications for per-user features.
