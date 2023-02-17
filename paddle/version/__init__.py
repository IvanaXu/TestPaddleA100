# THIS FILE IS GENERATED FROM PADDLEPADDLE SETUP.PY
#
full_version    = '0.0.0'
major           = '0'
minor           = '0'
patch           = '0'
rc              = '0'
cuda_version    = '11.2'
cudnn_version   = '8.1.1'
istaged         = False
commit          = 'b4a938840967f8e0d59a588a50a77f54f63961c8'
with_mkl        = 'ON'

__all__ = ['cuda', 'cudnn', 'show']

def show():
    """Get the version of paddle if `paddle` package if tagged. Otherwise, output the corresponding commit id.
    
    Returns:
        If paddle package is not tagged, the commit-id of paddle will be output.
        Otherwise, the following information will be output.

        full_version: version of paddle

        major: the major version of paddle

        minor: the minor version of paddle

        patch: the patch level version of paddle
        
        rc: whether it's rc version

        cuda: the cuda version of package. It will return `False` if CPU version paddle package is installed

        cudnn: the cudnn version of package. It will return `False` if CPU version paddle package is installed
    
    Examples:
        .. code-block:: python

            import paddle

            # Case 1: paddle is tagged with 2.2.0
            paddle.version.show()
            # full_version: 2.2.0
            # major: 2
            # minor: 2
            # patch: 0
            # rc: 0
            # cuda: '10.2'
            # cudnn: '7.6.5'

            # Case 2: paddle is not tagged
            paddle.version.show()
            # commit: cfa357e984bfd2ffa16820e354020529df434f7d
            # cuda: '10.2'
            # cudnn: '7.6.5'
    """
    if istaged:
        print('full_version:', full_version)
        print('major:', major)
        print('minor:', minor)
        print('patch:', patch)
        print('rc:', rc)
    else:
        print('commit:', commit)
    print('cuda:', cuda_version)
    print('cudnn:', cudnn_version)

def mkl():
    return with_mkl

def cuda():
    """Get cuda version of paddle package.

    Returns:
        string: Return the version information of cuda. If paddle package is CPU version, it will return False.
    
    Examples:
        .. code-block:: python

            import paddle

            paddle.version.cuda()
            # '10.2'

    """
    return cuda_version

def cudnn():
    """Get cudnn version of paddle package.

    Returns:
        string: Return the version information of cudnn. If paddle package is CPU version, it will return False.
    
    Examples:
        .. code-block:: python

            import paddle

            paddle.version.cudnn()
            # '7.6.5'

    """
    return cudnn_version
