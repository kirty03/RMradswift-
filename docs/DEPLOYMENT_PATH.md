# From course project to hospital use

None of the steps below are part of this project. They are what would have to happen next.

## How would a hospital use RadSwift, given the permissions needed?

1. **Permission to use data.** Research ethics board approval and a data-sharing agreement with the hospital.
   Patient images stay inside the hospital's own systems (PHIPA in Ontario; HIPAA is the U.S. equivalent).
2. **Retrospective test.** Run the model on the hospital's past chest X-rays and compare with the signed reports.
   Our model has only seen one public dataset, so its accuracy elsewhere is unknown.
3. **Silent trial.** The model drafts reports in the background. Nobody sees or uses them. Drafts are compared
   with what radiologists wrote.
4. **Supervised pilot.** A few radiologists see the drafts, edit them and sign. Measure real turnaround time,
   edit rates and errors.
5. **Regulation.** Software that drafts diagnostic reports is a medical device. In Canada it needs a Health
   Canada medical device licence; Health Canada has published pre-market guidance for machine-learning-enabled
   medical devices.
6. **Integration.** Connection to the image archive (PACS) and the radiology information system (RIS).

In every stage the radiologist reads the image, edits the draft and signs. The AI never sends a report itself.

## Why is this not already in every hospital?

It is beginning to appear, as drafts that a radiologist reviews:

- Northwestern Medicine built a draft-report system in-house and reported about a 15.5% average gain in
  documentation efficiency (JAMA Network Open, 2025).
- Aidoc "First Read" (chest X-ray report drafting) received an FDA Breakthrough Device Designation, announced
  25 June 2026. It is investigational, not cleared, and keeps clinician approval.
- Mosaic Clinical Technologies "Cognita Chest X-Ray" received the same designation in March 2026 and reported
  an 18% gain in interpretation efficiency.

What slows adoption: clinical accuracy (a fluent report can still be wrong), regulation, legal responsibility
for errors, integration cost, and the need to prove the system on each hospital's own patients.

Re-check each of these sources before quoting it on a slide.

## What our project adds and does not add

- Adds: a working fine-tuned draft-and-triage prototype on public data, measured honestly against baselines,
  and a simulation of the turnaround effect with every assumption written down.
- Does not add: evidence from a hospital, a clinically validated model, or a product.
