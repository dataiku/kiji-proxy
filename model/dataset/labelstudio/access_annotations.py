import os

from label_studio_sdk.client import LabelStudio

# Initalize the Label Studio client
base_url = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080")
api_key = os.environ["LABEL_STUDIO_API_KEY"]
project_id = os.environ["LABEL_STUDIO_PROJECT_ID"]

ls = LabelStudio(base_url=base_url, api_key=api_key)

tasks = ls.tasks.list(project=project_id)

# create a list of dictionaries to store all the annotations
all_annotations = []
for t in tasks:
    annotations = {}
    # we need to check to see if there are annotations for the task. If not, we skip it.
    if t.annotations:
        # Access the annotation results for each task.
        for ann in t.annotations[0]["result"]:
            # if the value field is present, we have an entity annotation.
            if "value" in ann:
                id = ann["id"]
                text = ann["value"]["text"]
                label = ann["value"]["labels"][0]
                annotations[id] = {"text": text, "label": label}
            # if the from_id field is present, we have a relation annotation. Add this to the from_id key.
            elif "from_id" in ann:
                from_text = annotations[ann["from_id"]]["text"]
                to_text = annotations[ann["to_id"]]["text"]
                relation = ann["type"]
                annotations[ann["from_id"]].update({"refers_to": to_text})
        all_annotations.append(annotations)
print(all_annotations)
