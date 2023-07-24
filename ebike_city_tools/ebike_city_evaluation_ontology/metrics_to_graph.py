import pandas as pd
from rdflib import Dataset, Literal, URIRef, XSD, RDF, RDFS
import uuid
import onto_manager as OM

EVALUATION_METRIC = 'EvaluationMetric'
METRIC_TYPE = 'MetricType'
EVALUATION_METHOD = 'EvaluationMethod'
EVALUATION_CRITERIA = 'EvaluationCriteria'
RESOLUTION_FEATURE = 'ResolutionFeature'
WEIGHTING_SYSTEM = 'WeightingSystem'
SCORING_SYSTEM = 'ScoringSystem'
MEASURING_MODE = 'MeasuringMode'
DESCRIPTION = 'Description'
UNIT = 'Unit'
UNIT_TYPE = 'UnitType'
PARTS = 'Parts'
FUNCTION = 'Function'
BUFFER = 'Buffer'
BUFFER_UNIT = 'BufferUnit'
WEIGHT = 'Weight'
COMMENT = 'Comment'


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
            self.dataset.add((metric_uri, RDF.type, metric_type, OM.NEMO_GRAPH))
            measure_uri = URIRef(OM.NEMO_GRAPH + str(uuid.uuid1()))
            self.dataset.add((measure_uri, RDF.type, OM.MEASURE, OM.NEMO_GRAPH))
            self.dataset.add((metric_uri, OM.HAS_VALUE, measure_uri, OM.NEMO_GRAPH))
            method_uri = URIRef(metrics.loc[metric, EVALUATION_METHOD])
            self.dataset.add((metric_uri, OM.USED_IN, method_uri, OM.NEMO_GRAPH))
            measuring_mode = URIRef(OM.PREFIX_NEMO + metrics.loc[metric, MEASURING_MODE])
            self.dataset.add((metric_uri, OM.MEASURED_WITH, measuring_mode, OM.NEMO_GRAPH))
            self.dataset.add((method_uri, RDF.type, OM.EVALUATION_METHOD, OM.NEMO_GRAPH))

            if not pd.isnull(metrics.loc[metric, COMMENT]):
                comment = Literal(str(metrics.loc[metric, COMMENT]), datatype=XSD.string)
                self.dataset.add((metric_uri, RDFS.comment, comment, OM.NEMO_GRAPH))
            if not pd.isnull(metrics.loc[metric, METRIC_TYPE]):
                general_metric_type = URIRef(OM.PREFIX_NEMO + metrics.loc[metric, METRIC_TYPE])
                self.dataset.add((measure_uri, RDFS.subClassOf, general_metric_type, OM.NEMO_GRAPH))
            if not pd.isnull(metrics.loc[metric, WEIGHTING_SYSTEM]):
                self.create_weighting_system_triples(metrics.loc[metric, WEIGHTING_SYSTEM], method_uri)
            if not pd.isnull(metrics.loc[metric, WEIGHT]):
                self.create_weight_triples(metrics.loc[metric, WEIGHT], metric_uri)
            if not pd.isnull(metrics.loc[metric, SCORING_SYSTEM]):
                self.create_scoring_system_triples(metrics.loc[metric, SCORING_SYSTEM], metric_uri)
            if not pd.isnull(metrics.loc[metric, UNIT]):
                self.create_unit_triples(metrics.loc[metric, UNIT], metrics.loc[metric, UNIT_TYPE], measure_uri)
            if not pd.isnull(metrics.loc[metric, PARTS]):
                self.create_metric_part_triples(metrics.loc[metric, PARTS].split(','), metric_uri)
            if not pd.isnull(metrics.loc[metric, EVALUATION_CRITERIA]):
                self.create_criteria_triples(metrics.loc[metric, EVALUATION_CRITERIA].split(','), metric_uri)
            if not pd.isnull(metrics.loc[metric, RESOLUTION_FEATURE]):
                self.create_resolution_triples(metrics.loc[metric, RESOLUTION_FEATURE], metric_uri)
            if not pd.isnull(metrics.loc[metric, FUNCTION]):
                self.create_aggregate_function_triples(metrics.loc[metric, FUNCTION], metric_uri)
            if not pd.isnull(metrics.loc[metric, BUFFER]):
                buffer_list = str(metrics.loc[metric, BUFFER])
                self.create_buffer_triples(buffer_list.split(';'), metrics.loc[metric, BUFFER_UNIT], metric_uri)

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

    def create_measuring_mode_triples(self, measuring_mode, metric_uri):
        """
        creates measuring mode triples that are linked to evaluation metric.
        Args:
            measuring_mode: the way metric was measured, e.g. objective or subjective.
            metric_uri: relevant method uri
        """
        measuring_mode_uri = URIRef(OM.PREFIX_NEMO + measuring_mode)
        self.dataset.add((metric_uri, OM.MEASURED_WITH, measuring_mode_uri, OM.NEMO_GRAPH))
        self.dataset.add((measuring_mode_uri, RDF.type, OM.MEASURING_MODE, OM.NEMO_GRAPH))

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

    def create_scoring_system_triples(self, scoring_system, metric_uri):
        """
        creates evaluation scale triples linked to the particular metric.
        Args:
            scoring_system: an evaluation scale, e.g. 0-100 percent or 1-5 score, for reference check OM ontology.
            metric_uri: relevant metric uri.
        """
        scoring_system = URIRef(OM.PREFIX_OM + scoring_system)
        self.dataset.add((metric_uri, OM.NORMALISED_TO_SCORING_SYSTEM, scoring_system, OM.NEMO_GRAPH))

    def create_aggregate_function_triples(self, function, metric_uri):
        """
        creates aggregate function triples. It is used if metric value for example is an average.
        Args:
            function: a function can be applied to a quantity, like imn, max, average, for reference check OM ontology.
            metric_uri: relevant metric uri.
        """
        function_uri = URIRef(OM.PREFIX_OM + function)
        self.dataset.add((function_uri, RDF.type, OM.FUNCTION, OM.NEMO_GRAPH))
        self.dataset.add((metric_uri, OM.HAS_AGGREGATE_FUNCTION, function_uri, OM.NEMO_GRAPH))

    def create_unit_triples(self, units, unit_type, measure_uri):
        """
        creates unit triples for the relevant metric.
        Args:
            units: refer to OM ontology.
            unit_type: more general unit type compared to unit instance.
            measure_uri: relevant metric uri.
        """
        unit_type_uri = URIRef(OM.PREFIX_OM + unit_type)
        if len(units) == 3:
            pass
        elif len(units) == 2:
            combined_unit_uri = URIRef(OM.PREFIX_OM + units[0] + 'Per' + units[1])
            numerator_uri = URIRef(OM.PREFIX_OM + units[0])
            denominator_uri = URIRef(OM.PREFIX_OM + units[1])
            self.dataset.add((measure_uri, OM.HAS_UNIT, combined_unit_uri, OM.NEMO_GRAPH))
            self.dataset.add((combined_unit_uri, RDF.type, unit_type_uri, OM.NEMO_GRAPH))
            self.dataset.add((combined_unit_uri, OM.HAS_NUMERATOR, numerator_uri, OM.NEMO_GRAPH))
            self.dataset.add((combined_unit_uri, OM.HAS_DENOMINATOR, denominator_uri, OM.NEMO_GRAPH))
        else:
            unit_uri = URIRef(OM.PREFIX_OM + units[0])
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
            self.dataset.add((metric_uri, OM.COMPOSED_OF, part_uri, OM.NEMO_GRAPH))

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
        self.dataset.add((metric_uri, OM.AGGREGATED_ON, res_feature_uri, OM.NEMO_GRAPH))

    def create_criteria_triples(self, criterias, metric_uri):
        """
        creates triples defining bike network goodness criteria for which relevant metric is used.
        Args:
            criterias: A bike network goodness criteria, like safety, efficiency, quality, attractiveness.
            metric_uri: relevant metric uri.
        """
        for criteria in criterias:
            criteria_uri = URIRef(OM.PREFIX_NEMO + str(uuid.uuid1()))
            criteria_class = URIRef(OM.PREFIX_NEMO + criteria)
            self.dataset.add((criteria_uri, RDF.type, criteria_class, OM.NEMO_GRAPH))
            self.dataset.add((metric_uri, OM.MEASURES, criteria_uri, OM.NEMO_GRAPH))

    def create_buffer_triples(self, buffers, buffer_unit, metric_uri):
        """
        creates triples to define a buffered area around network element (edge or node)
        within which certain metrics are estimated.
        Args:
            buffers: distance from the edge or node.
            buffer_unit: unit that defines buffer extent, e.g. metre, minute.
            metric_uri: relevant metric uri.
        """
        for buffer in buffers:
            buffer_uri = URIRef(OM.NEMO_GRAPH + str(uuid.uuid1()))
            measure_uri = URIRef(OM.NEMO_GRAPH + str(uuid.uuid1()))
            buffer_distance = Literal(str(buffer), datatype=XSD.integer)
            self.dataset.add((metric_uri, OM.MEASURED_WITHIN_BUFFER, buffer_uri, OM.NEMO_GRAPH))
            self.dataset.add((buffer_uri, OM.HAS_VALUE, measure_uri, OM.NEMO_GRAPH))
            self.dataset.add((measure_uri, OM.HAS_UNIT, URIRef(OM.PREFIX_OM + buffer_unit), OM.NEMO_GRAPH))
            self.dataset.add((measure_uri, OM.HAS_NUMERIC_VALUE, buffer_distance, OM.NEMO_GRAPH))

    def write_triples(self, out_dir):

        with open(out_dir + "output_metrics.nq", mode="w") as file:
            file.write(self.dataset.serialize(format='nquads'))


root_dir = 'C:/Users/agrisiute/OneDrive - ETH Zurich/Desktop/nemo/'
out_dir = root_dir + 'output/'
input_dir = root_dir + 'input/'

input_file = input_dir + 'metrics.xlsx'
metric_df = pd.DataFrame(pd.read_excel(input_file))

metric_dataset = TripleDataset()

metric_dataset.create_metric_triples(metric_df)
metric_dataset.write_triples(out_dir)

