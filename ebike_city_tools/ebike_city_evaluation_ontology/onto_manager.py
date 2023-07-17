from rdflib import URIRef

PREFIX_NEMO = 'http://www.ebikecityevaluationtool.com/ontology/nemo#'
PREFIX_OM = 'http://www.ontology-of-units-of-measure.org/resource/om-2/'
NEMO_GRAPH = URIRef('http://www.ebikecityevaluationtool.com/ontology/nemo/metrics/')

# Nemo Entities
EVALUATION_CRITERIA = URIRef(PREFIX_NEMO + 'EvaluationCriteria')
EVALUATION_SCALE = URIRef(PREFIX_NEMO + 'EvaluationScale')
EVALUATION_METHOD = URIRef(PREFIX_NEMO + 'EvaluationMethod')
WEIGHTING_SYSTEM = URIRef(PREFIX_NEMO + 'WeightingSystem')
WEIGHT = URIRef(PREFIX_NEMO + 'Weight')
RESOLUTION_FEATURE = URIRef(PREFIX_NEMO + 'ResolutionFeature')
EVALUATION_METRIC = URIRef(PREFIX_NEMO + 'EvaluationMetric')
COMPOSITE_METRIC = URIRef(PREFIX_NEMO + 'CompositeMetric')
MEASURING_MODE = URIRef(PREFIX_NEMO + 'MeasuringMode')
OBJECTIVE = URIRef(PREFIX_NEMO + 'Objective')
SUBJECTIVE = URIRef(PREFIX_NEMO + 'Subjective')

CONTEXTUAL_METRIC = URIRef(PREFIX_NEMO + 'ContextualMetric')
INFRASTRUCTURAL_METRIC = URIRef(PREFIX_NEMO + 'InfrastructuralMetric')
MORPHOLOGICAL_METRIC = URIRef(PREFIX_NEMO + 'MorphologicalMetric')
MULTIMODAL_METRIC = URIRef(PREFIX_NEMO + 'MultimodalMetric')
TOPOLOGICAL_METRIC = URIRef(PREFIX_NEMO + 'TopologicalMetric')

# Nemo properties
IS_NORMALISED_TO_SCALE = URIRef(PREFIX_NEMO + 'IsNormalisedToScale')
IS_AGGREGATED_ON = URIRef(PREFIX_NEMO + 'isAggregatedOn')
IS_COMPOSED_OF = URIRef(PREFIX_NEMO + 'isComposedOf')
HAS_WEIGHTING_SYSTEM = URIRef(PREFIX_NEMO + 'hasWeightingSystem')
IS_MEASURED_BY = URIRef(PREFIX_NEMO + 'isMeasuredBy')
MEASURED_WITH = URIRef(PREFIX_NEMO + 'measuredWith')
IS_USED_IN = URIRef(PREFIX_NEMO + 'isUsedIn')
MEASURES = URIRef(PREFIX_NEMO + 'measures')
USES_METRIC = URIRef(PREFIX_NEMO + 'usesMetric')
PART_OF = URIRef(PREFIX_NEMO + 'partOf')
WEIGHTED_BY = URIRef(PREFIX_NEMO + 'weightedBy')
HAS_DATA_SOURCE = URIRef(PREFIX_NEMO + 'hasDataSource')
HAS_CREATION_DATE = URIRef(PREFIX_NEMO + 'hasCreationDate')
IS_OBJECTIVE = URIRef(PREFIX_NEMO + 'isObjective')
IS_PERCEIVED = URIRef(PREFIX_NEMO + 'isPerceived')
IS_MEASURED_WITHIN_BUFFER = URIRef(PREFIX_NEMO + 'isMeasuredWithinBuffer')

# Om Entities
FUNCTION = URIRef(PREFIX_OM + 'Function')
MEASURE = URIRef(PREFIX_OM + 'Measure')
QUANTITY = URIRef(PREFIX_OM + 'Quantity')
UNIT = URIRef(PREFIX_OM + 'Unit')
METRE = URIRef(PREFIX_OM + 'metre')

# Om Properties
HAS_AGGREGATE_FUNCTION = URIRef(PREFIX_OM + 'hasAggregateFunction')
HAS_UNIT = URIRef(PREFIX_OM + 'hasUnit')
HAS_VALUE = URIRef(PREFIX_OM + 'hasValue')
HAS_NUMERIC_VALUE = URIRef(PREFIX_OM + 'hasNumericValue')


