# CellForge SDK-First + Showcase + Pilot Master Plan

## Product Objective
Ship a stable, fast, and easy-to-integrate CellForge SDK for UE5/C++ teams, prove value with pilot customers, and convert pilots into repeatable paid usage.

## Single Best Path (Selected)
UE5-first SDK product with a high-quality preset showcase and two paid pilots.

Why this is the best path:
- fastest route from R&D to revenue
- highest signal for real demand in this niche
- easiest story for buyers: visual impact + low integration risk

## Target Segment (Initial)
- small and mid-size UE5 studios
- technical art and VFX teams
- gameplay teams building destructible or regenerative systems

## Positioning
"Production-ready CellForge runtime for real-time self-healing and reactive materials in UE5, with C-API and hot-reload."

## Product Pillars
1. Stable SDK Core
2. Instant Visual Value (Showcase)
3. Pilot-to-Production Delivery Motion

## 90-Day Roadmap

### Phase 1 (Weeks 1-3): SDK Hardening
Goal: remove integration and stability risk.

Deliverables:
- freeze C-API v1 baseline
- add strict error contract and docs per function
- add capabilities query API (backend, limits, feature flags)
- add structured runtime logs and optional host callback sink
- add crash-safe smoke tests and 8h soak test
- add deterministic replay mode for QA
- publish binary artifacts and versioned changelog

Exit criteria:
- no crash in 8h soak test on target matrix
- all public API calls covered by integration tests
- successful load, update, render, destroy across test scenarios

### Phase 2 (Weeks 2-6): Showcase Pack
Goal: produce visible "wow" and clear use cases.

Deliverables:
- 5 production-grade presets:
  - Self-Heal Armor
  - Organic Growth
  - Corruption Spread
  - Impact Scar + Recovery
  - Energy Membrane
- one UE5 sample map per preset
- one 60-90s reel and five 15-20s clips
- preset controls: quality profile, intensity, damage radius, recovery speed

Exit criteria:
- each preset demonstrates a unique gameplay or VFX use case
- all presets run at target FPS profile on reference hardware

### Phase 3 (Weeks 5-12): Pilot Motion
Goal: convert technical readiness into commercial proof.

Deliverables:
- 20 targeted outreach contacts
- 8 technical discovery calls
- 3 pilot proposals
- 2 paid pilots signed
- 2 pilot completion reports with ROI evidence

Exit criteria:
- at least one pilot converts to paid continuation
- at least one reusable case study approved

## Engineering Quality Gates (Tier-1)

### API and ABI
- semantic versioning for C-API
- ABI compatibility checks for each release
- no breaking changes without major version bump

### Performance
- target profiles (1080p):
  - Fast: >= 120 FPS
  - Balanced: >= 90 FPS
  - Cinematic: >= 60 FPS
- frame-time budget tracking in sample app

### Stability
- zero memory leaks in create/destroy loops
- zero data races in TSAN-safe host call patterns
- graceful handling of invalid pointers and bad inputs

### Packaging
- release assets include:
  - nca_engine.dll
  - nca_engine.lib
  - C header
  - version manifest
  - UE5 sample integration package

## KPI Dashboard
- time-to-first-frame in UE5: <= 2 hours
- integration success rate from docs only: >= 80%
- pilot close rate from qualified calls: >= 25%
- pilot to paid conversion: >= 50%
- support tickets per integration: <= 3 median

## Commercial Packaging

### Indie
- low-cost license
- limited presets
- community support

### Studio
- annual seat or project license
- full preset pack
- priority support

### Enterprise
- custom SLA
- custom effects and optimization support
- roadmap influence and dedicated onboarding

## Risks and Mitigations
- Risk: effect quality looks repetitive
  - Mitigation: diversify training targets and preset-specific palettes
- Risk: integration friction in UE5
  - Mitigation: plugin template + sample map + 30-minute onboarding guide
- Risk: unstable behavior under stress
  - Mitigation: soak test, deterministic mode, bounded update rules
- Risk: long sales cycle
  - Mitigation: short paid pilot with fixed scope and quick ROI narrative

## Immediate Execution (Next 10 Working Days)
1. Freeze C-API v1 and publish function-by-function docs.
2. Add capabilities and stats endpoint to FFI.
3. Build one polished UE5 demo map with two presets.
4. Record first showcase clip.
5. Prepare pilot proposal template and outreach list.
6. Schedule first 5 discovery calls.

## Definition of Done: "Stable Product v1"
- reproducible build pipeline for SDK artifacts
- passing CI for core API integration tests
- passing 8h soak stability test
- UE5 sample project loads and runs without manual code edits
- at least one pilot-ready package signed off by external tester
