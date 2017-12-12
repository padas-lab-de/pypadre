from mongoengine import Document, SequenceField, FileField, StringField

class Dataset(Document):
    data_id = SequenceField()
    name = StringField()
    metadata = FileField()
    data = FileField()
    target = FileField()
