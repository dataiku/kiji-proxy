from label_studio_sdk.client import LabelStudio

# Initalize the Label Studio client 
# NOTE: what's the URL when hosted in the uv env?

ls = LabelStudio(
    base_url="http://localhost:8080",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA3MzAxNDc5NywiaWF0IjoxNzY1ODE0Nzk3LCJqdGkiOiIxZDA5NWE5NDg5YTY0YWJiYTliZDM5ODNmMjk1NWYyOSIsInVzZXJfaWQiOiIxIn0.FOiuCILa5N7QOvmrIZ9fs-JjFnbt-7wSqE4DtqE8zus"
)

tasks = ls.tasks.list(project=3)

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
                annotations[id] = {
                    "text": text,
                    "label": label
                }
            # if the from_id field is present, we have a relation annotation. Add this to the from_id key.
            elif "from_id" in ann:
                from_text = annotations[ann["from_id"]]["text"]
                to_text = annotations[ann["to_id"]]["text"]
                relation = ann["type"]
                annotations[ann["from_id"]].update({"refers_to": to_text})
        all_annotations.append(annotations) 
print(all_annotations)