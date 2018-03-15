from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import mock

from rasa_nlu.project import Project


def test_dynamic_load_model_with_exists_model():
    MODEL_NAME = 'model_name'

    def mocked_init(*args, **kwargs):
        return None

    with mock.patch.object(Project, "__init__", mocked_init):
        project = Project()

        project._models = (MODEL_NAME, )

        result = project._dynamic_load_model(MODEL_NAME)

        assert result == MODEL_NAME


def test_dynamic_load_model_with_refresh_exists_model():
    MODEL_NAME = 'model_name'

    def mocked_init(*args, **kwargs):
        return None

    def mocked_search_for_models(self):
        self._models = (MODEL_NAME, )

    with mock.patch.object(Project, "__init__", mocked_init):
        with mock.patch.object(Project, '_search_for_models', mocked_search_for_models):
            project = Project()

            project._models = ()

            result = project._dynamic_load_model(MODEL_NAME)

            assert result == MODEL_NAME


def test_dynamic_load_model_with_refresh_not_exists_model():
    LATEST_MODEL_NAME = 'latest_model_name'

    def mocked_init(*args, **kwargs):
        return None

    def mocked_search_for_models(self):
        pass

    def mocked_latest_trained_project_model(self):
        return LATEST_MODEL_NAME

    with mock.patch.object(Project, "__init__", mocked_init):
        with mock.patch.object(Project, "_search_for_models", mocked_search_for_models):
            with mock.patch.object(Project, "_latest_trained_project_model", mocked_latest_trained_project_model):
                project = Project()

                project._models = ()

                result = project._dynamic_load_model('model_name')

                assert result == LATEST_MODEL_NAME


def test_dynamic_load_model_with_model_is_none():
    LATEST_MODEL_NAME = 'latest_model_name'

    def mocked_init(*args, **kwargs):
        return None

    def mocked_search_for_models(self):
        pass

    def mocked_latest_trained_project_model(self):
        return LATEST_MODEL_NAME

    with mock.patch.object(Project, "__init__", mocked_init):
        with mock.patch.object(Project, "_search_for_models", mocked_search_for_models):
            with mock.patch.object(Project, "_latest_trained_project_model", mocked_latest_trained_project_model):
                project = Project()

                project._models = ()

                result = project._dynamic_load_model(None)

                assert result == LATEST_MODEL_NAME


def test_latest_model_and_unload():
    PROJECT = 'project_name'
    LATEST_USED_MODEL = 'latest_used_model_name'
    LATEST_TRAINED_MODEL = 'latest_trained_model_name'
    MODELS = {LATEST_USED_MODEL: 'dummy used model',
              LATEST_TRAINED_MODEL: 'dummy trained model'}

    def mocked_init(self, *args, **kwargs):
        from threading import Lock
        self._latest_used_project_model = LATEST_USED_MODEL
        self._writer_lock = Lock()
        self._project = PROJECT

    def mocked_search_for_models(self):
        pass

    with mock.patch.object(Project, "__init__", mocked_init):
        with mock.patch.object(Project, "_search_for_models", mocked_search_for_models):
            project = Project()

            project._models = MODELS

            assert project._latest_used_project_model == LATEST_USED_MODEL

            project._set_latest_model_and_unload(LATEST_TRAINED_MODEL)

            assert project._models == {LATEST_TRAINED_MODEL: 'dummy trained model'}

            assert project._latest_used_project_model == LATEST_TRAINED_MODEL
