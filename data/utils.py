class UnableToCrop(Exception):
    def __init__(self, message, im_path=''):
        super().__init__(message + ' ' + im_path)
        self.im_path = im_path
