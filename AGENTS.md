# Repository Map For Agents

This repository is `sebosimo/XCBenz_Data`, the public data/fetch repo.

Related repo:

- `sebosimo/XCBenz_Temps`: private Streamlit app repo. It contains app/UI code and private thermal-model code.
- `sebosimo/XCBenz_Data`: this repo. GitHub Actions fetch ICON data and force-push generated `.nc` files to the `data` branch.

The live app reads this repo's `data` branch through raw GitHub URLs.

Rules of thumb:

- Production fetcher, `locations.json`, manifest, cache layout, and packed NetCDF changes belong here.
- UI/rendering changes belong in `XCBenz_Temps`.
- If a change affects both generated data and app reading/display, update both repos together.
- Keep private app code and thermal-model code out of this public repo.
- Do not manually force-push `main`; the workflow is responsible for force-pushing the generated `data` branch.
