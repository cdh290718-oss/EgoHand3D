# Third-Party Notices and Claim Boundary

This project contains application-layer code for first-person hand
reconstruction workflows. It integrates with external model code, model assets,
datasets, and scientific Python libraries.

## Application Code and Provenance Status

The original 17 application Python files contained 3214 physical lines. That
number is a size statistic, **not verified original-source ownership**. A local
comparison found exact copies of four scripts from the local WiLoR directory,
a closely adapted demo entry point, and reused helper functions. The owner
reports mixed, currently uncertain provenance for the additional local scripts.
Those files must be traced before making an original-source claim.

The current extension adds application modules for input preflight, persistent
tasks, explicit MANO interchange and consistency checks, sequence processing,
quality diagnostics, evaluation and static reports. The implementation and
its tests are maintained in this repository; this does not establish ownership
of dependencies or change the upstream model/data licenses. Preserve the
actual development record and review provenance for registration materials.

The WiLoR neural network, detection backend and MANO model remain external
components. Runtime tests and generated outputs do not establish originality
or automatically authorize redistribution.

## Excluded From Original-Source Claim

The following are not claimed as original self-developed source code:

- `wilor/`: WiLoR-based model, dataset, loss, renderer, and utility code.
- `pretrained_models/`: pretrained checkpoint and detector assets.
- `mano_data/`: MANO model data.
- datasets and WebDataset tar files.
- example photos and videos.
- generated outputs, logs, reports, and caches.
- software-copyright registration drafts and official templates.

## Public-Release Caution

Before making this repository public, verify whether the local `wilor/` source
may be redistributed under its upstream license. If redistribution is not
permitted or the license is unclear, do one of the following:

1. remove `wilor/` from the public repository and document how users should
   install it from the upstream project;
2. use a Git submodule that points to the upstream repository;
3. keep this repository private.

Pretrained weights, MANO data, datasets, and example images should not be
uploaded unless their licenses explicitly permit redistribution.

## Dependency Categories

Runtime and development dependencies include, but are not limited to:

- Python
- PyTorch
- PyTorch Lightning
- Hydra
- OpenCV
- NumPy
- Ultralytics
- Trimesh
- Pyrender
- WebDataset

Each dependency remains governed by its own license.

## Suggested Citation and Attribution

If this project is published for research or engineering reuse, include
attribution for the upstream hand reconstruction model and any datasets used for
training or evaluation. Keep upstream citation text in the README or a dedicated
`CITATION.cff` file once the exact upstream references are confirmed.

## Published snapshot and submission scope

The exact application-source listing is `registration/source_scope.json`. The 13 listed files are submitted in full (28 pages); model code, legacy adapters, tests and experiment configurations are excluded from this added-application claim. The repository retains legacy scripts for workflow compatibility with the provenance limitations above. The source audit is `registration/source_overlap_audit.json`.

The upstream WiLoR license inspected is CC BY-NC-ND 4.0; see `LICENSE_SCOPE.md`. This repository does not apply a blanket license to third-party or uncertain-origin material.
