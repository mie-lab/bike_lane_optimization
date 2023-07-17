import pandas as pd
from rdflib import Dataset, Literal, URIRef, XSD, RDF
import uuid
import onto_manager as OM

EVALUATION_METRIC = 'EvaluationMetric'
EVALUATION_METHOD = 'EvaluationMethod'
EVALUATION_CRITERIA = 'EvaluationCriteria'
RESOLUTION_FEATURE = 'ResolutionFeature'
WEIGHTING_SYSTEM = 'WeightingSystem'
EVALUATION_SCALE = 'EvaluationScale'
MEASURING_MODE = 'MeasuringMode'
DATA_SOURCE = 'DataSource'
UNIT = 'Unit'
PARTS = 'Parts'
FUNCTION = 'Function'
BUFFER = 'Buffer'
UNIT_TYPE = 'UnitType'
WEIGHT = 'Weight'

class TripleDataset:

    def __init__(self):
        self.dataset = Dataset()

    def create_metric_triples(self, metrics):
        """
        creates triples to represent bike network evaluation metric used in an existing evaluation method (paper).
        Args:
            metrics: a metric with various associated properties described in a paper.
        """

        for metric in metrics.index:

            metric_uri = URIRef(OM.NEMO_GRAPH + str(uuid.uuid1()))
            metric_type = URIRef(OM.PREFIX_NEMO + metrics.loc[metric, EVALUATION_METRIC])
            measure_uri = URIRef(OM.NEMO_GRAPH + str(uuid.uuid1()))
            method_uri = URIRef(metrics.loc[metric, EVALUATION_METHOD])
            measuring_mode = URIRef(OM.PREFIX_NEMO + metrics.loc[metric, MEASURING_MODE])

            self.dataset.add((metric_uri, RDF.type, metric_type, OM.NEMO_GRAPH))
            self.dataset.add((measure_uri, RDF.type, OM.MEASURE, OM.NEMO_GRAPH))
            self.dataset.add((metric_uri, OM.HAS_VALUE, measure_uri, OM.NEMO_GRAPH))
            self.dataset.add((metric_uri, OM.IS_USED_IN, method_uri, OM.NEMO_GRAPH))
            self.dataset.add((metric_uri, OM.MEASURED_WITH, measuring_mode, OM.NEMO_GRAPH))
            self.dataset.add((method_uri, RDF.type, OM.EVALUATION_METHOD, OM.NEMO_GRAPH))

            if not pd.isnull(metrics.loc[metric, DATA_SOURCE]):
                self.dataset.add((metric_uri, OM.HAS_DATA_SOURCE, URIRef(metrics.loc[metric, DATA_SOURCE])))
            if not pd.isnull(metrics.loc[metric, WEIGHTING_SYSTEM]):
                self.create_weighting_system_triples(metrics.loc[metric, WEIGHTING_SYSTEM], method_uri)
            if not pd.isnull(metrics.loc[metric, WEIGHT]):
                self.create_weight_triples(metrics.loc[metric, WEIGHT], metric_uri)
            if not pd.isnull(metrics.loc[metric, EVALUATION_SCALE]):
                self.create_evaluation_scale_triples(metrics.loc[metric, EVALUATION_SCALE], metric_uri)
            if not pd.isnull(metrics.loc[metric, UNIT]):
                self.create_unit_triples(metrics.loc[metric, UNIT], metrics.loc[metric, UNIT_TYPE], measure_uri)
            if not pd.isnull(metrics.loc[metric, PARTS]):
                self.create_metric_part_triples(metrics.loc[metric, PARTS].split(','), metric_uri)
            if not pd.isnull(metrics.loc[metric, EVALUATION_CRITERIA]):
                self.create_criteria_triples(metrics.loc[metric, EVALUATION_CRITERIA], metric_uri)
            if not pd.isnull(metrics.loc[metric, RESOLUTION_FEATURE]):
                self.create_resolution_triples(metrics.loc[metric, RESOLUTION_FEATURE], metric_uri)
            if not pd.isnull(metrics.loc[metric, FUNCTION]):
                self.create_aggr_function_triples(metrics.loc[metric, FUNCTION], metric_uri)
            if not pd.isnull(metrics.loc[metric, BUFFER]):
                self.create_buffer_triples(metrics.loc[metric, BUFFER], metric_uri)

    def create_weight_triples(self, weight, metric_uri):
        """
        creates weight triples that are linked to the particular metric.
        Args:
            weight: the weight value form the research paper
            metric_uri: relevant metric uri.
        """
        weight_value = Literal(str(weight), datatype=XSD.float)
        weight_uri = URIRef(OM.NEMO_GRAPH + str(uuid.uuid1()))
        weight_measure = URIRef(OM.NEMO_GRAPH + str(uuid.uuid1()))
        self.dataset.add((metric_uri, OM.WEIGHTED_BY, weight_uri, OM.NEMO_GRAPH))
        self.dataset.add((weight_uri, RDF.type, OM.WEIGHT, OM.NEMO_GRAPH))
        self.dataset.add((weight_uri, OM.HAS_VALUE, weight_measure, OM.NEMO_GRAPH))
        self.dataset.add((weight_measure, OM.HAS_NUMERIC_VALUE, weight_value, OM.NEMO_GRAPH))

    def create_weighting_system_triples(self, weighting, method_uri):
        """
        creates weighting system triples that are linked to evaluation method
        Args:
            weighting: weighting system name
            method_uri: relevant method uri
        """
        weighting_system_uri = URIRef(OM.PREFIX_NEMO + weighting)
        self.dataset.add((method_uri, OM.HAS_WEIGHTING_SYSTEM, weighting_system_uri, OM.NEMO_GRAPH))
        self.dataset.add((weighting_system_uri, RDF.type, OM.WEIGHTING_SYSTEM, OM.NEMO_GRAPH))

    def create_evaluation_scale_triples(self, evaluation_scale, metric_uri):
        """
        creates evaluation scale triples linked to the prticular metric.
        Args:
            evaluation_scale: an evluaiton scale, e.g. 0-100 percent or 1-5 score, for reference check OM ontology.
            metric_uri: relevant metric uri.
        """
        evaluation_scale = URIRef(OM.PREFIX_OM + evaluation_scale)
        self.dataset.add((metric_uri, OM.IS_NORMALISED_TO_SCALE, evaluation_scale, OM.NEMO_GRAPH))

    def create_aggr_function_triples(self, function, metric_uri):
        """
        creates aggregate function triples. It is used if metric value for example is an average.
        Args:
            function: a function can be applied to a quantity, like imn, max, average, for reference check OM ontology.
            metric_uri: relevant metric uri.
        """
        function_uri = URIRef(OM.PREFIX_OM + function)
        self.dataset.add((function_uri, RDF.type, OM.FUNCTION, OM.NEMO_GRAPH))
        self.dataset.add((metric_uri, OM.HAS_AGGREGATE_FUNCTION, function_uri, OM.NEMO_GRAPH))

    def create_unit_triples(self, unit, unit_type, measure_uri):
        """
        creates unit triples for the relevant metric.
        Args:
            unit: refer to OM ontology.
            unit_type: more general unit type compared to unit instance.
            measure_uri: relevant metric uri.
        """
        unit_uri = URIRef(OM.PREFIX_OM + unit)
        unit_type_uri = URIRef(OM.PREFIX_OM + unit_type)
        self.dataset.add((measure_uri, OM.HAS_UNIT, unit_uri, OM.NEMO_GRAPH))
        self.dataset.add((unit_uri, RDF.type, unit_type_uri, OM.NEMO_GRAPH))

    def create_metric_part_triples(self, parts, metric_uri):
        """
        creates part triples for metrics that is composed of multiple metrics.
        Args:
            parts: other metrics from which the relevant metric is composed.
            metric_uri: relevant metric uri.
        """
        for part in parts:
            part_uri = URIRef(OM.NEMO_GRAPH + str(uuid.uuid1()))
            self.dataset.add((part_uri, RDF.type, URIRef(OM.PREFIX_NEMO + part), OM.NEMO_GRAPH))
            self.dataset.add((metric_uri, OM.IS_COMPOSED_OF, part_uri, OM.NEMO_GRAPH))

    def create_resolution_triples(self, resolution_feature, metric_uri):
        """
        creates triples that define on which spatial elements the metric value is aggregated (calculated).
        Args:
            resolution_feature: a spatial feature,on which the relavant metric is calculated, e.g. network, edge, route
            metric_uri: relevant metric uri.
        """
        res_feature_uri = URIRef(OM.PREFIX_NEMO + str(uuid.uuid1()))
        res_feature_class = URIRef(OM.PREFIX_NEMO + resolution_feature)
        self.dataset.add((res_feature_uri, RDF.type, res_feature_class, OM.NEMO_GRAPH))
        self.dataset.add((metric_uri, OM.IS_AGGREGATED_ON, res_feature_uri, OM.NEMO_GRAPH))

    def create_criteria_triples(self, criteria, metric_uri):
        """
        creates triples defining bike network goodness criteria for which relevant metric is used.
        Args:
            criteria: A goodness criteria, like safety, efficiency, quality, attractiveness.
            metric_uri: relevant metric uri.
        """
        criteria_uri = URIRef(OM.PREFIX_NEMO + str(uuid.uuid1()))
        criteria_class = URIRef(OM.PREFIX_NEMO + criteria)
        self.dataset.add((criteria_uri, RDF.type, criteria_class, OM.NEMO_GRAPH))
        self.dataset.add((metric_uri, OM.MEASURES, criteria_uri, OM.NEMO_GRAPH))

    def create_buffer_triples(self, buffer, metric_uri):
        """
        creates triples to define a buffered area around network element (edge or node) within which certain metrics are estimated.
        Args:
            buffer: distance from the edge or node.
            metric_uri: relevant metric uri.
        """
        buffer_uri = URIRef(OM.NEMO_GRAPH + str(uuid.uuid1()))
        measure_uri = URIRef(OM.NEMO_GRAPH + str(uuid.uuid1()))
        buffer_distance = Literal(str(buffer), datatype=XSD.integer)
        buffer_unit = OM.METRE
        self.dataset.add((metric_uri, OM.IS_MEASURED_WITHIN_BUFFER, buffer_uri, OM.NEMO_GRAPH))
        self.dataset.add((buffer_uri, OM.HAS_VALUE, measure_uri, OM.NEMO_GRAPH))
        self.dataset.add((measure_uri, OM.HAS_UNIT, buffer_unit, OM.NEMO_GRAPH))
        self.dataset.add((measure_uri, OM.HAS_NUMERIC_VALUE, buffer_distance, OM.NEMO_GRAPH))

    def write_triples(self, out_dir):

        with open(out_dir + "output_metrics.nq", mode="w") as file:
            file.write(self.dataset.serialize(format='nquads'))


root_dir = 'C:/Users/agrisiute/Desktop/'
out_dir = root_dir + 'output/'
input_dir = root_dir + 'input/'

input_file = input_dir + 'test.xlsx'
metric_df = pd.DataFrame(pd.read_excel(input_file))

metric_dataset = TripleDataset()

metric_dataset.create_metric_triples(metric_df)
metric_dataset.write_triples(out_dir)

