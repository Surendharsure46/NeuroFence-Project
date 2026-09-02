CONTROLLED TEST ARTEFACT — DO NOT DISTRIBUTE OR DEPLOY
======================================================

This model directory has been DELIBERATELY MODIFIED by NeuroFence for the sole
purpose of validating a backdoor detector against known ground truth.

The modification scales selected input-embedding rows so that a specific token
produces an amplified activation signature. It is an activation marker only:
it teaches the model no behaviour and does not alter what the model says.

This is a test fixture, equivalent in spirit to an EICAR antivirus test file.
It is not a functional backdoor and must not be used as one.

See backdoor_manifest.json for the exact modification and how to reverse it.
