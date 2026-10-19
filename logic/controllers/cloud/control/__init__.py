"""Hosted control planes: Firebase (Cloud Firestore) and Supabase (Postgres).

Both are :class:`~logic.controllers.cloud.control.documents.DocumentControlPlane`
over a provider adapter, and both pass the CR-01 contract suite. Their SDKs
(``google-cloud-firestore``, ``supabase``) are imported only by
``control_plane_from_uri``; the classes take an injected client.
"""
