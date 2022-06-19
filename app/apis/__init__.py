# Copyright (c) 2022, Ko Sugawara
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1.  Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
# 2.  Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==============================================================================
from flask_restx import Api

from elephant import __version__

from .dataset import api as dataset
from .download import api as download
from .flow import api as flow
from .params import api as params
from .seg import api as seg
from .state import api as state
from .upload import api as upload


def init_api(app):
    """
    Initialize Api and returns it.

    Parameters
    ----------
    app : Flask
        Flask object.

    Returns
    -------
    api : flask_restx.Api
        Api object.

    """
    api = Api(
        app=app,
        version=__version__,
        title='ELEPHANT',
        description='ELEPHANT API',
    )

    api.add_namespace(dataset, path='/dataset')
    api.add_namespace(download, path='/download')
    api.add_namespace(flow, path='/flow')
    api.add_namespace(params, path='/params')
    api.add_namespace(seg, path='/seg')
    api.add_namespace(state, path='/state')
    api.add_namespace(upload, path='/upload')

    return api
