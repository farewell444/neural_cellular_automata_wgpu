# Sprint Backlog (6 Weeks)

## Sprint 1: API and Reliability Baseline
- [ ] Lock C-API signatures for v1 baseline
- [ ] Add nca_engine_get_capabilities
- [ ] Add nca_engine_get_runtime_stats
- [ ] Add centralized error code map and docs
- [ ] Add create/destroy stress test (10000 iterations)
- [ ] Add invalid input fuzz tests for FFI layer

Acceptance:
- [ ] All tests pass in CI
- [ ] No crashes in stress tests

## Sprint 2: Performance and Diagnostics
- [ ] Add per-frame timing metrics (update and render)
- [ ] Add quality profiles (Fast/Balanced/Cinematic)
- [ ] Add profiling overlay in sample app
- [ ] Add GPU capability detection and fallback paths
- [ ] Add hot-reload event diagnostics

Acceptance:
- [ ] Stable frame time at target profile on reference GPU
- [ ] Diagnostic info available via API

## Sprint 3: Showcase Presets (Part 1)
- [ ] Implement Self-Heal Armor preset package
- [ ] Implement Corruption Spread preset package
- [ ] Add preset serialization format
- [ ] Add preset switch API call
- [ ] Add one UE5 sample map for these presets

Acceptance:
- [ ] Both presets usable from UE5 sample without code changes

## Sprint 4: Showcase Presets (Part 2)
- [ ] Implement Organic Growth preset package
- [ ] Implement Impact Scar + Recovery preset package
- [ ] Implement Energy Membrane preset package
- [ ] Add polished visual tuning controls
- [ ] Capture final reel and short clips

Acceptance:
- [ ] 5 preset showcase complete
- [ ] Visual reel exported

## Sprint 5: Pilot Readiness
- [ ] Create pilot SOW template
- [ ] Create integration checklist and onboarding guide
- [ ] Build one-click sample package archive
- [ ] Create issue triage and SLA response workflow
- [ ] Prepare pricing and contract skeleton

Acceptance:
- [ ] External tester can integrate in <= 2 hours

## Sprint 6: Pilot Execution
- [ ] Run discovery with first cohort
- [ ] Sign at least 2 pilot agreements
- [ ] Deliver pilot integrations
- [ ] Collect FPS and production-impact metrics
- [ ] Produce one case-study draft

Acceptance:
- [ ] At least one pilot ready for paid continuation
