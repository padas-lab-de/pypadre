#!/usr/bin/env python

from enum import Enum

class PaDREOntology:

	class SubClassesUnit(Enum):
		Count = "http://www.padre-lab.eu/onto/Count"
		Length = "http://www.padre-lab.eu/onto/Length"
		Weight = "http://www.padre-lab.eu/onto/Weight"

	class SubClassesDatum(Enum):
		Boolean = "http://www.padre-lab.eu/onto/Boolean"
		Character = "http://www.padre-lab.eu/onto/Character"
		Double = "http://www.padre-lab.eu/onto/Double"
		Float = "http://www.padre-lab.eu/onto/Float"
		Integer = "http://www.padre-lab.eu/onto/Integer"
		Long = "http://www.padre-lab.eu/onto/Long"
		Number = "http://www.padre-lab.eu/onto/Number"

	class SubClassesMeasurement(Enum):
		Interval = "http://www.padre-lab.eu/onto/Interval"
		Nominal = "http://www.padre-lab.eu/onto/Nominal"
		Ordinal = "http://www.padre-lab.eu/onto/Ordinal"
		Ratio = "http://www.padre-lab.eu/onto/Ratio"

	class SubClassesDataset(Enum):
		Graph = "http://www.padre-lab.eu/onto/Graph"
		Multivariat = "http://www.padre-lab.eu/onto/Multivariat"
		Timeseries = "http://www.padre-lab.eu/onto/Timeseries"

	class SubClassesExperiment(Enum):
		Classification = "http://www.padre-lab.eu/onto/Classification"
		Regression = "http://www.padre-lab.eu/onto/Regression"
		BinaryClassification = "http://www.padre-lab.eu/onto/Binary"
		Clustering = "http://www.padre-lab.eu/onto/Clustering"
		FlatClustering = "http://www.padre-lab.eu/onto/Flat"
		HIerarchicalClustering = "http://www.padre-lab.eu/onto/Hierarchical"
		MulticlassClassification = "http://www.padre-lab.eu/onto/Multiclass"
		MultilabelClassification = "http://www.padre-lab.eu/onto/Multilabel"
		Scoring = "http://www.padre-lab.eu/onto/Scoring"
		SequenceLabeling = "http://www.padre-lab.eu/onto/SequenceLabeling"
		SimilarityLearning = "http://www.padre-lab.eu/onto/SimilarityLearning"
		Transformation = "http://www.padre-lab.eu/onto/Transformation"

	def __init__(self):
		self.subClassesUnit = self.SubClassesUnit()
		self.subClassesDatum = self.SubClassesDatum()
		self.subClassesMeasurement = self.SubClassesMeasurement()
		self.subClassesDataset = self.SubClassesDataset()
		self.subClassesExperiment = self.SubClassesExperiment()
