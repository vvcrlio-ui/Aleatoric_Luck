# Generated FFCWS schemas

Run `python FFCWS/adapter/prepare.py` from the repository root. The adapter
writes one authoritative schema per strategy/outcome and one shared canonical
feature-universe definition per strategy into this directory.

All schemas use the provider's official external test split and
`train_pool_screened` feature universes. The generated schema and universe files
are versioned; raw and derived tables remain ignored.
