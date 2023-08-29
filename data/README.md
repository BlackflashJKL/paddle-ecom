## Data Format Description

The corpus consists of English and Chinese data, in ECOB-EN and ECOB-ZH directory separately. Each directory contains train/val/test.doc.json and  train/val/test.ann.json of the following formats.

In train/dev/test.doc.json, each JSON instance represents a document.

一个event对应多个doc，

```json
{
    "Descriptor": {
        "event_id": (int) event_id,
        "text": "Event descriptor."
    },
    "Doc": {
        "doc_id": (int) doc_id,
        "title": "Title of document.",
        "content": [
            {
                "sent_idx": 0,
                "sent_text": "Raw text of the first sentence."
            },
            {
                "sent_idx": 1,
                "sent_text": "Raw text of the second sentence."
            },
            ...
            {
                "sent_idx": n-1,
                "sent_text": "Raw text of the (n-1)th sentence."
            }
        ]
    }
}
```

In train/dev/test.ann.json, each JSON instance represents an opinion extracted from documents.

```json
{
    "Descriptor": {
        "event_id": (int) event_id,
        "text": "Event descriptor."
    },
    "Doc": {
        "doc_id": (int) doc_id,
        "title": "Title of document.",
        "content": [
            {
                "sent_idx": 0,
                "sent_text": "Raw text of the first sentence."
            },
            {
                "sent_idx": 1,
                "sent_text": "Raw text of the second sentence."
            },
            ...
            {
                "sent_idx": n-1,
                "sent_text": "Raw text of the (n-1)th sentence."
            }
        ]
    }
}
```





