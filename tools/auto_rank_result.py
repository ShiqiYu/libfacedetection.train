"""auto rank evaluation results."""
import os


class AutoRank(object):

    def __init__(self, rank_path) -> None:
        self.rank_path = rank_path
        self.sep = 18

        if os.path.exists(rank_path):
            with open(rank_path, 'r') as f:
                file_content = f.readlines()
            self.content = self.analysis_rank(file_content)
        else:
            self.content = []

    def analysis_rank(self, file_content):
        content = []
        length = len(file_content)
        assert length % 5 == 0
        for idx in range(0, length, 5):
            per_content = {}
            per_content['config'] = file_content[idx][self.sep:-1]
            per_content['weight'] = file_content[idx + 1][self.sep:-1]
            per_content['score_nms_thresh'] = [
                float(x)
                for x in file_content[idx + 2][self.sep + 1:-2].split(',')
            ]
            per_content['APS'] = [
                float(x)
                for x in file_content[idx + 3][self.sep + 1:-2].split(',')
            ]
            content.append(per_content)
        return content

    def add_content(self, it):
        tmp_content = {}
        config = it.get('config', None)
        assert config is not None
        tmp_content['config'] = config

        weight = it.get('weight', None)
        assert weight is not None
        tmp_content['weight'] = weight

        score_nms_thresh = it.get('score_nms_thresh', None)
        assert score_nms_thresh is not None
        tmp_content['score_nms_thresh'] = score_nms_thresh

        aps = it.get('APS', None)
        assert aps is not None
        assert isinstance(aps, list)
        assert len(aps) == 3
        tmp_content['APS'] = aps

        self.content.append(tmp_content)

    def sort_content(self, reverse=True):
        self.content.sort(key=lambda x: x['APS'][2], reverse=reverse)

    def store_content(self):
        s = ''
        for it in self.content:
            for k, v in it.items():
                s += f'{k.ljust(self.sep - 1)}:{v}\n'
            s += '\n'
        with open(self.rank_path, 'w') as f:
            f.write(s)

    def update(self, it=None, reverse=True):
        if it is not None:
            self.add_content(it)
        self.sort_content(reverse)
        self.store_content()


if __name__ == '__main__':
    x = AutoRank('./eval.log')
    x.update()

    # x.update({'config': "dasdasda", 'weight': "dfghdfghdfh",
    # 'APS': [231, 231, 22], 'score_nms_thresh':[0.3, 0.45]})
