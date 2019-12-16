class Attribute(dict):

    def __init__(self, name, measurementLevel=None, unit=None,
                 description=None, defaultTargetAttribute=False, context=None, index=None, type=None, nullable=True):

        if context is None:
            context={}
        dict.__init__(self, name=name, measurementLevel=measurementLevel,
                      unit=unit, description=description, defaultTargetAttribute=defaultTargetAttribute,
                      context=context, index=index, type=type, nullable=nullable)

    @property
    def name(self):
        if "name" in self:
            return self["name"]
        else:
            self["name"] = None
            return None

    @property
    def index(self):
        if "index" in self:
            return self["index"]
        else:
            self["index"] = None
            return None

    @property
    def measurementLevel(self):
        if "measurementLevel" in self:
            return self["measurementLevel"]
        else:
            self["measurementLevel"] = ""
            return self["measurementLevel"]

    @property
    def unit(self):
        if "unit" in self:
            return self["unit"]
        else:
            self["unit"] = ""
            return self["unit"]

    @property
    def description(self):
        if "description" in self:
            return self["description"]
        else:
            self["description"] = ""
            return self["description"]

    @property
    def defaultTargetAttribute(self):
        if "defaultTargetAttribute" in self:
            return self["defaultTargetAttribute"]
        else:
            self["defaultTaretAttribute"] = False
            return False

    @property
    def context(self):
        if "context" in self:
            return self["context"]
        else:
            self["context"] = dict()
            return self["context"]

    def __str__(self):
        return self.name
# + "(" + str(self.measurementLevel) + ")"

    def __repr__(self):
        if "graph_role" in self.context:
            return self.name + "(" + self.context["graph_role"] + ")"
        else:
            return str(self["name"])
# + "(" + str(self["measurementLevel"]) + "/" + str(self["unit"]) + ")"