class Singleton(type):
  _instances = {}

  def __call__(self, *args, **kwargs):
    if self not in self._instances:
      self._instances[self] = super().__call__(*args, **kwargs)
    return self._instances[self]
