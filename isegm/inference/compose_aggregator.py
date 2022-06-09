class ComposeAggregator:
    def __init__(self, itypes_aggregators, exclude_reset=None):
        self.itypes_aggregators = itypes_aggregators
        self.input_type = list(itypes_aggregators.keys())
        self.exclude_reset = exclude_reset or []

    def reset(self):
        for itype, aggr in self.itypes_aggregators.items():
            if itype not in self.exclude_reset:
                aggr.reset()

    def make_next(self, pred_mask, input_type=None):
        if input_type is None:
            for itype, aggr in self.itypes_aggregators.items():
                aggr.make_next(pred_mask)
        else:
            self.itypes_aggregators[input_type].make_next(pred_mask)

    def get_interactions(self, limit=None):
        interaction_dict = {}
        for itype, aggr in self.itypes_aggregators.items():
            interaction_dict[itype] = aggr.get_interactions(limit)
        return interaction_dict

    @property
    def interactions_dict(self):
        interactions_dict = {}
        for itype, aggr in self.itypes_aggregators.items():
            interactions_dict[itype] = aggr.interactions_list
        return interactions_dict

    def set_index_offset(self, index_offset):
        for itype, aggr in self.itypes_aggregators.items():
            aggr.set_index_offset(index_offset)

    def get_state(self, input_type=None):
        if input_type is not None:
            return {input_type: self.itypes_aggregators[input_type].get_state()}
        states = {}
        for itype, aggr in self.itypes_aggregators.items():
            states[itype] = aggr.get_state()
        return states

    def set_state(self, states):
        for itype, aggr in self.itypes_aggregators.items():
            if itype in states.keys():
                aggr.set_state(states[itype])

    def __getitem__(self, item):
        return self.itypes_aggregators[item]
