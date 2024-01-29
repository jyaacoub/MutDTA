from enum import Enum, EnumMeta
import sys
def _make_class_unpicklable(cls):
    """
    Make the given class un-picklable.
    """
    def _break_on_call_reduce(self, proto):
        raise TypeError('%r cannot be pickled' % self)
    cls.__reduce_ex__ = _break_on_call_reduce
    cls.__module__ = '<unknown>'
    
class MetaEnum(EnumMeta):
    def __repr__(self) -> str:
        return str(list(map(lambda c: c.value, self)))
    
    def __contains__(cls, item):
        return item in cls.__members__
    
    def _create_(cls, class_name, names, *, module=None, qualname=None, type=None, start=1):
        """
        Convenience method to create a new Enum class.

        `names` can be:

        * A string containing member names, separated either with spaces or
          commas.  Values are incremented by 1 from `start`.
        * An iterable of member names.  Values are incremented by 1 from `start`.
        * An iterable of (member name, value) pairs.
        * A mapping of member name -> value pairs.
        """
        #### NOTE: INJECTED CODE TO CHANGE DEFAULT BEHAVIOR OF ENUM ###########
        if isinstance(names, list) and names and isinstance(names[0], str):
            names = [(i,i) for i in names]
        #######################################################################
        
        metacls = cls.__class__
        bases = (cls, ) if type is None else (type, cls)
        _, first_enum = cls._get_mixins_(cls, bases)
        classdict = metacls.__prepare__(class_name, bases)

        # special processing needed for names?
        if isinstance(names, str):
            names = names.replace(',', ' ').split()
        if isinstance(names, (tuple, list)) and names and isinstance(names[0], str):
            original_names, names = names, []
            last_values = []
            for count, name in enumerate(original_names):
                value = first_enum._generate_next_value_(name, start, count, last_values[:])
                last_values.append(value)
                names.append((name, value))

        # Here, names is either an iterable of (name, value) or a mapping.
        for item in names:
            if isinstance(item, str):
                member_name, member_value = item, names[item]
            else:
                member_name, member_value = item
            classdict[member_name] = member_value
        enum_class = metacls.__new__(metacls, class_name, bases, classdict)

        # TODO: replace the frame hack if a blessed way to know the calling
        # module is ever developed
        if module is None:
            try:
                module = sys._getframe(2).f_globals['__name__']
            except (AttributeError, ValueError, KeyError):
                pass
        if module is None:
            _make_class_unpicklable(enum_class)
        else:
            enum_class.__module__ = module
        if qualname is not None:
            enum_class.__qualname__ = qualname

        return enum_class
    
    def __getitem__(self, index):
        return self.list()[index]

class CustomEnum(Enum, metaclass=MetaEnum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))