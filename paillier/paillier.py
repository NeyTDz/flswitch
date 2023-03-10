import random
import sys
import math

from paillier.integer_arithmetic import IntegerArithmetic
import numpy as np


class Cryptosystem(object):
    def encrypt(self, plaintext):
        pass

    def decrypt(self, ciphertext):
        pass

    @staticmethod
    def generate(*args):
        """
        Generate an instance
        :param args:
        :return:
        """
        pass


class Ciphertext(object):
    def __init__(self, message):
        self._message = message

    @property
    def message(self):
        return self._message

    def __eq__(self, other):
        if self._message == other.message:
            return True
        else:
            return False

    def __hash__(self):
        """
        Must be explicitly inherited
        :return:
        """
        return hash(str(self._message))

class AsymmetricCryptosystem(Cryptosystem):
    def __init__(self, public_key, private_key=None):
        """
        :param public_key: dict
        :param private_key: dict
        """
        self._public_key = public_key
        self._private_key = private_key

class CPaillier(AsymmetricCryptosystem):
    """
    This crypto preserves additive homomorphism
    """

    def __init__(self, public_key, private_key=None):
        super(CPaillier, self).__init__(public_key, private_key)

    def encrypt(self, plaintext):
        return self._public_key.encrypt(plaintext)

    def decrypt(self, ciphertext):
        return self._private_key.decrypt(ciphertext)

    @property
    def public_key(self):
        return self._public_key

    @staticmethod
    def generate(key_size=1024):
        p = q = n = None
        n_len = 0
        while n_len != key_size:
            p = IntegerArithmetic.getprimeover(key_size // 2)
            q = p
            while q == p:
                q = IntegerArithmetic.getprimeover(key_size // 2)
            n = p * q
            n_len = n.bit_length()

        public_key = PaillierPublicKey(n)
        private_key = PaillierPrivateKey(public_key, p, q)

        return CPaillier(public_key, private_key)


class PaillierPublicKey(object):
    """Contains a public key and associated encryption methods.
    Args:
      n (int): the modulus of the public key - see Paillier's paper.
    Attributes:
      g (int): part of the public key - see Paillier's paper.
      n (int): part of the public key - see Paillier's paper.
      nsquare (int): :attr:`n` ** 2, stored for frequent use.
      max_int (int): Maximum int that may safely be stored. This can be
        increased, if you are happy to redefine "safely" and lower the
        chance of detecting an integer overflow.
    """

    def __init__(self, n):
        self.g = n + 1
        self.n = n
        self.nsquare = n * n
        self.max_int = n // 3 - 1

    def __repr__(self):
        publicKeyHash = hex(hash(self))[2:]
        return "<PaillierPublicKey {}>".format(publicKeyHash[:10])

    def __eq__(self, other):
        return self.n == other.n

    def __hash__(self):
        return hash(self.n)

    def raw_encrypt(self, plaintext, r_value=None):
        """Paillier encryption of a positive integer plaintext < :attr:`n`.
        You probably should be using :meth:`encrypt` instead, because it
        handles positive and negative ints and floats.
        Args:
          plaintext (int): a positive integer < :attr:`n` to be Paillier
            encrypted. Typically this is an encoding of the actual
            number you want to encrypt.
          r_value (int): obfuscator for the ciphertext; by default (i.e.
            r_value is None), a random value is used.
        Returns:
          int: Paillier encryption of plaintext.
        Raises:
          TypeError: if plaintext is not an int.
        """
        if not isinstance(plaintext, int):
            raise TypeError('Expected int type plaintext but got: %s' %
                            type(plaintext))

        if self.n - self.max_int <= plaintext < self.n:
            # Very large plaintext, take a sneaky shortcut using inverses
            neg_plaintext = self.n - plaintext  # = abs(plaintext - nsquare)
            neg_ciphertext = (self.n * neg_plaintext + 1) % self.nsquare
            nude_ciphertext = IntegerArithmetic.invert(neg_ciphertext, self.nsquare)
        else:
            # we chose g = n + 1, so that we can exploit the fact that
            # (n+1)^plaintext = n*plaintext + 1 mod n^2
            nude_ciphertext = (self.n * plaintext + 1) % self.nsquare

        r = r_value or self.get_random_lt_n()
        obfuscator = IntegerArithmetic.powmod(r, self.n, self.nsquare)

        return (nude_ciphertext * obfuscator) % self.nsquare

    def get_random_lt_n(self):
        """Return a cryptographically random number less than :attr:`n`"""
        return random.SystemRandom().randrange(1, self.n)

    def encrypt(self, value, precision=None, r_value=None):
        """Encode and Paillier encrypt a real number *value*.
        Args:
          value: an int or float to be encrypted.
            If int, it must satisfy abs(*value*) < :attr:`n`/3.
            If float, it must satisfy abs(*value* / *precision*) <<
            :attr:`n`/3
            (i.e. if a float is near the limit then detectable
            overflow may still occur)
          precision (float): Passed to :meth:`EncodedNumber.encode`.
            If *value* is a float then *precision* is the maximum
            **absolute** error allowed when encoding *value*. Defaults
            to encoding *value* exactly.
          r_value (int): obfuscator for the ciphertext; by default (i.e.
            if *r_value* is None), a random value is used.
        Returns:
          PaillierCiphertext: An encryption of *value*.
        Raises:
          ValueError: if *value* is out of range or *precision* is so
            high that *value* is rounded to zero.
        """

        if isinstance(value, EncodedNumber):
            encoding = value
        else:
            encoding = EncodedNumber.encode(self, value, precision)

        return self.encrypt_encoded(encoding, r_value)

    def encrypt_encoded(self, encoding, r_value):
        """Paillier encrypt an encoded value.
        Args:
          encoding: The EncodedNumber instance.
          r_value (int): obfuscator for the ciphertext; by default (i.e.
            if *r_value* is None), a random value is used.
        Returns:
          PaillierCiphertext: An encryption of *value*.
        """
        # If r_value is None, obfuscate in a call to .obfuscate() (below)
        obfuscator = r_value or 1
        ciphertext = self.raw_encrypt(encoding.encoding, r_value=obfuscator)
        encrypted_number = PaillierCiphertext(self, ciphertext, encoding.exponent)
        if r_value is None:
            encrypted_number.obfuscate()
        return encrypted_number


class PaillierPrivateKey(object):
    """Contains a private key and associated decryption method.
    Args:
      public_key (:class:`PaillierPublicKey`): The corresponding public
        key.
      p (int): private secret - see Paillier's paper.
      q (int): private secret - see Paillier's paper.
    Attributes:
      public_key (PaillierPublicKey): The corresponding public
        key.
      p (int): private secret - see Paillier's paper.
      q (int): private secret - see Paillier's paper.
      psquare (int): p^2
      qsquare (int): q^2
      p_inverse (int): p^-1 mod q
      hp (int): h(p) - see Paillier's paper.
      hq (int): h(q) - see Paillier's paper.
    """

    def __init__(self, public_key, p, q):
        if not p * q == public_key.n:
            raise ValueError('given public key does not match the given p and q.')
        if p == q:
            # check that p and q are different, otherwise we can't compute p^-1 mod q
            raise ValueError('p and q have to be different')
        self.public_key = public_key
        if q < p:  # ensure that p < q.
            self.p = q
            self.q = p
        else:
            self.p = p
            self.q = q
        self.psquare = self.p * self.p

        self.qsquare = self.q * self.q
        self.p_inverse = IntegerArithmetic.invert(self.p, self.q)
        self.hp = self.h_function(self.p, self.psquare)
        self.hq = self.h_function(self.q, self.qsquare)

    @staticmethod
    def from_totient(public_key, totient):
        """given the totient, one can factorize the modulus
        The totient is defined as totient = (p - 1) * (q - 1),
        and the modulus is defined as modulus = p * q
        Args:
          public_key (PaillierPublicKey): The corresponding public
            key
          totient (int): the totient of the modulus
        Returns:
          the :class:`PaillierPrivateKey` that corresponds to the inputs
        Raises:
          ValueError: if the given totient is not the totient of the modulus
            of the given public key
        """
        p_plus_q = public_key.n - totient + 1
        p_minus_q = IntegerArithmetic.isqrt(p_plus_q * p_plus_q - public_key.n * 4)
        q = (p_plus_q - p_minus_q) // 2
        p = p_plus_q - q
        if not p * q == public_key.n:
            raise ValueError('given public key and totient do not match.')
        return PaillierPrivateKey(public_key, p, q)

    def __repr__(self):
        pub_repr = repr(self.public_key)
        return "<PaillierPrivateKey for {}>".format(pub_repr)

    def decrypt(self, encrypted_number):
        """Return the decrypted & decoded plaintext of *encrypted_number*.
        Uses the default :class:`EncodedNumber`, if using an alternative encoding
        scheme, use :meth:`decrypt_encoded` or :meth:`raw_decrypt` instead.
        Args:
          encrypted_number (PaillierCiphertext): an
            :class:`EncryptedNumber` with a public key that matches this
            private key.
        Returns:
          the int or float that `EncryptedNumber` was holding. N.B. if
            the number returned is an integer, it will not be of type
            float.
        Raises:
          TypeError: If *encrypted_number* is not an
            :class:`EncryptedNumber`.
          ValueError: If *encrypted_number* was encrypted against a
            different key.
        """
        encoded = self.decrypt_encoded(encrypted_number)
        return encoded.decode()

    def decrypt_encoded(self, encrypted_number, Encoding=None):
        """Return the :class:`EncodedNumber` decrypted from *encrypted_number*.
        Args:
          encrypted_number (PaillierCiphertext): an
            :class:`EncryptedNumber` with a public key that matches this
            private key.
          Encoding (class): A class to use instead of :class:`EncodedNumber`, the
            encoding used for the *encrypted_number* - used to support alternative
            encodings.
        Returns:
          :class:`EncodedNumber`: The decrypted plaintext.
        Raises:
          TypeError: If *encrypted_number* is not an
            :class:`EncryptedNumber`.
          ValueError: If *encrypted_number* was encrypted against a
            different key.
        """
        if not isinstance(encrypted_number, PaillierCiphertext):
            raise TypeError('Expected encrypted_number to be an PaillierCiphertext'
                            ' not: %s' % type(encrypted_number))

        if self.public_key != encrypted_number.public_key:
            raise ValueError('encrypted_number was encrypted against a '
                             'different key!')

        if Encoding is None:
            Encoding = EncodedNumber

        encoded = self.raw_decrypt(encrypted_number.ciphertext(be_secure=False))
        return Encoding(self.public_key, encoded,
                        encrypted_number.exponent)

    def raw_decrypt(self, ciphertext):
        """Decrypt raw ciphertext and return raw plaintext.
        Args:
          ciphertext (int): (usually from :meth:`PaillierCiphertext.ciphertext()`)
            that is to be Paillier decrypted.
        Returns:
          int: Paillier decryption of ciphertext. This is a positive
          integer < :attr:`public_key.n`.
        Raises:
          TypeError: if ciphertext is not an int.
        """
        if not isinstance(ciphertext, int):
            raise TypeError('Expected ciphertext to be an int, not: %s' %
                            type(ciphertext))

        decrypt_to_p = self.l_function(IntegerArithmetic.powmod(ciphertext, self.p - 1, self.psquare),
                                       self.p) * self.hp % self.p
        decrypt_to_q = self.l_function(IntegerArithmetic.powmod(ciphertext, self.q - 1, self.qsquare),
                                       self.q) * self.hq % self.q
        return self.crt(decrypt_to_p, decrypt_to_q)

    def h_function(self, x, xsquare):
        """Computes the h-function as defined in Paillier's paper page 12,
        'Decryption using Chinese-remaindering'.
        """
        return IntegerArithmetic.invert(self.l_function(IntegerArithmetic.powmod(self.public_key.g, x - 1, xsquare), x),
                                        x)

    def l_function(self, x, p):
        """Computes the L function as defined in Paillier's paper. That is: L(x,p) = (x-1)/p"""
        return (x - 1) // p

    def crt(self, mp, mq):
        """The Chinese Remainder Theorem as needed for decryption. Returns the solution modulo n=pq.
        Args:
           mp(int): the solution modulo p.
           mq(int): the solution modulo q.
       """
        u = (mq - mp) * self.p_inverse % self.q
        return mp + (u * self.p)

    def __eq__(self, other):
        return self.p == other.p and self.q == other.q

    def __hash__(self):
        return hash((self.p, self.q))


class EncodedNumber(object):
    """Represents a float or int encoded for Paillier encryption.
    For end users, this class is mainly useful for specifying precision
    when adding/multiplying an :class:`EncryptedNumber` by a scalar.
    If you want to manually encode a number for Paillier encryption,
    then use :meth:`encode`, if de-serializing then use
    :meth:`__init__`.
    .. note::
        If working with other Paillier libraries you will have to agree on
        a specific :attr:`BASE` and :attr:`LOG2_BASE` - inheriting from this
        class and overriding those two attributes will enable this.
    Notes:
      Paillier encryption is only defined for non-negative integers less
      than :attr:`PaillierPublicKey.n`. Since we frequently want to use
      signed integers and/or floating point numbers (luxury!), values
      should be encoded as a valid integer before encryption.
      The operations of addition and multiplication [1]_ must be
      preserved under this encoding. Namely:
      1. Decode(Encode(a) + Encode(b)) = a + b
      2. Decode(Encode(a) * Encode(b)) = a * b
      for any real numbers a and b.
      Representing signed integers is relatively easy: we exploit the
      modular arithmetic properties of the Paillier scheme. We choose to
      represent only integers between
      +/-:attr:`~PaillierPublicKey.max_int`, where `max_int` is
      approximately :attr:`~PaillierPublicKey.n`/3 (larger integers may
      be treated as floats). The range of values between `max_int` and
      `n` - `max_int` is reserved for detecting overflows. This encoding
      scheme supports properties #1 and #2 above.
      Representing floating point numbers as integers is a harder task.
      Here we use a variant of fixed-precision arithmetic. In fixed
      precision, you encode by multiplying every float by a large number
      (e.g. 1e6) and rounding the resulting product. You decode by
      dividing by that number. However, this encoding scheme does not
      satisfy property #2 above: upon every multiplication, you must
      divide by the large number. In a Paillier scheme, this is not
      possible to do without decrypting. For some tasks, this is
      acceptable or can be worked around, but for other tasks this can't
      be worked around.
      In our scheme, the "large number" is allowed to vary, and we keep
      track of it. It is:
        :attr:`BASE` ** :attr:`exponent`
      One number has many possible encodings; this property can be used
      to mitigate the leak of information due to the fact that
      :attr:`exponent` is never encrypted.
      For more details, see :meth:`encode`.
    .. rubric:: Footnotes
    ..  [1] Technically, since Paillier encryption only supports
      multiplication by a scalar, it may be possible to define a
      secondary encoding scheme `Encode'` such that property #2 is
      relaxed to:
        Decode(Encode(a) * Encode'(b)) = a * b
      We don't do this.
    Args:
      public_key (PaillierPublicKey): public key for which to encode
        (this is necessary because :attr:`~PaillierPublicKey.max_int`
        varies)
      encoding (int): The encoded number to store. Must be positive and
        less than :attr:`~PaillierPublicKey.max_int`.
      exponent (int): Together with :attr:`BASE`, determines the level
        of fixed-precision used in encoding the number.
    Attributes:
      public_key (PaillierPublicKey): public key for which to encode
        (this is necessary because :attr:`~PaillierPublicKey.max_int`
        varies)
      encoding (int): The encoded number to store. Must be positive and
        less than :attr:`~PaillierPublicKey.max_int`.
      exponent (int): Together with :attr:`BASE`, determines the level
        of fixed-precision used in encoding the number.
    """
    BASE = 16
    """Base to use when exponentiating. Larger `BASE` means
    that :attr:`exponent` leaks less information. If you vary this,
    you'll have to manually inform anyone decoding your numbers.
    """
    LOG2_BASE = math.log(BASE, 2)
    FLOAT_MANTISSA_BITS = sys.float_info.mant_dig

    def __init__(self, public_key, encoding, exponent):
        self.public_key = public_key
        self.encoding = encoding
        self.exponent = exponent

    @classmethod
    def encode(cls, public_key, scalar, precision=None, max_exponent=None):
        """Return an encoding of an int or float.
        This encoding is carefully chosen so that it supports the same
        operations as the Paillier crypto.
        If *scalar* is a float, first approximate it as an int, `int_rep`:
            scalar = int_rep * (:attr:`BASE` ** :attr:`exponent`),
        for some (typically negative) integer exponent, which can be
        tuned using *precision* and *max_exponent*. Specifically,
        :attr:`exponent` is chosen to be equal to or less than
        *max_exponent*, and such that the number *precision* is not
        rounded to zero.
        Having found an integer representation for the float (or having
        been given an int `scalar`), we then represent this integer as
        a non-negative integer < :attr:`~PaillierPublicKey.n`.
        Paillier homomorphic arithemetic works modulo
        :attr:`~PaillierPublicKey.n`. We take the convention that a
        number x < n/3 is positive, and that a number x > 2n/3 is
        negative. The range n/3 < x < 2n/3 allows for overflow
        detection.
        Args:
          public_key (PaillierPublicKey): public key for which to encode
            (this is necessary because :attr:`~PaillierPublicKey.n`
            varies).
          scalar: an int or float to be encrypted.
            If int, it must satisfy abs(*value*) <
            :attr:`~PaillierPublicKey.n`/3.
            If float, it must satisfy abs(*value* / *precision*) <<
            :attr:`~PaillierPublicKey.n`/3
            (i.e. if a float is near the limit then detectable
            overflow may still occur)
          precision (float): Choose exponent (i.e. fix the precision) so
            that this number is distinguishable from zero. If `scalar`
            is a float, then this is set so that minimal precision is
            lost. Lower precision leads to smaller encodings, which
            might yield faster computation.
          max_exponent (int): Ensure that the exponent of the returned
            `EncryptedNumber` is at most this.
        Returns:
          EncodedNumber: Encoded form of *scalar*, ready for encryption
          against *public_key*.
        """
        if np.abs(scalar) < 1e-200:
            scalar = 0
        # Calculate the maximum exponent for desired precision
        if precision is None:
            if isinstance(scalar, int) or isinstance(scalar, np.int16) or \
                    isinstance(scalar, np.int32) or isinstance(scalar, np.int64):
                prec_exponent = 0
            elif isinstance(scalar, float) or isinstance(scalar, np.float16) \
                    or isinstance(scalar, np.float32) or isinstance(scalar, np.float64):
                # Encode with *at least* as much precision as the python float
                # What's the base-2 exponent on the float?
                bin_flt_exponent = math.frexp(scalar)[1]

                # What's the base-2 exponent of the least significant bit?
                # The least significant bit has value 2 ** bin_lsb_exponent
                bin_lsb_exponent = bin_flt_exponent - cls.FLOAT_MANTISSA_BITS

                # What's the corresponding base BASE exponent? Round that down.
                prec_exponent = math.floor(bin_lsb_exponent / cls.LOG2_BASE)
            else:
                raise TypeError("Don't know the precision of type %s."
                                % type(scalar))
        else:
            prec_exponent = math.floor(math.log(precision, cls.BASE))

        # Remember exponents are negative for numbers < 1.
        # If we're going to store numbers with a more negative
        # exponent than demanded by the precision, then we may
        # as well bump up the actual precision.
        if max_exponent is None:
            exponent = prec_exponent
        else:
            exponent = min(max_exponent, prec_exponent)

        int_rep = int(round(scalar * pow(cls.BASE, -exponent)))

        if abs(int_rep) > public_key.max_int:
            raise ValueError('Integer needs to be within +/- %d but got %d'
                             % (public_key.max_int, int_rep))

        # Wrap negative numbers by adding n
        return cls(public_key, int_rep % public_key.n, exponent)

    def decode(self):
        """Decode plaintext and return the result.
        Returns:
          an int or float: the decoded number. N.B. if the number
            returned is an integer, it will not be of type float.
        Raises:
          OverflowError: if overflow is detected in the decrypted number.
        """
        if self.encoding >= self.public_key.n:
            # Should be mod n
            raise ValueError('Attempted to decode corrupted number')
        elif self.encoding <= self.public_key.max_int:
            # Positive
            mantissa = self.encoding
        elif self.encoding >= self.public_key.n - self.public_key.max_int:
            # Negative
            mantissa = self.encoding - self.public_key.n
        else:
            raise OverflowError('Overflow detected in decrypted number')

        return mantissa * pow(self.BASE, self.exponent)

    def decrease_exponent_to(self, new_exp):
        """Return an `EncodedNumber` with same value but lower exponent.
        If we multiply the encoded value by :attr:`BASE` and decrement
        :attr:`exponent`, then the decoded value does not change. Thus
        we can almost arbitrarily ratchet down the exponent of an
        :class:`EncodedNumber` - we only run into trouble when the encoded
        integer overflows. There may not be a warning if this happens.
        This is necessary when adding :class:`EncodedNumber` instances,
        and can also be useful to hide information about the precision
        of numbers - e.g. a protocol can fix the exponent of all
        transmitted :class:`EncodedNumber` to some lower bound(s).
        Args:
          new_exp (int): the desired exponent.
        Returns:
          EncodedNumber: Instance with the same value and desired
            exponent.
        Raises:
          ValueError: You tried to increase the exponent, which can't be
            done without decryption.
        """
        if new_exp > self.exponent:
            raise ValueError('New exponent %i should be more negative than'
                             'old exponent %i' % (new_exp, self.exponent))
        factor = pow(self.BASE, self.exponent - new_exp)
        new_enc = self.encoding * factor % self.public_key.n
        return self.__class__(self.public_key, new_enc, new_exp)


class PaillierCiphertext(Ciphertext):
    def __init__(self, public_key, ciphertext, exponent=0):
        super(PaillierCiphertext, self).__init__({
            'public_key': public_key,
            'ciphertext': ciphertext,
            'exponent': exponent,
            'is_obfuscated': False
        })

        if isinstance(self.ciphertext, PaillierCiphertext):
            raise TypeError('ciphertext should be an integer')
        if not isinstance(self.public_key, PaillierPublicKey):
            raise TypeError('public_key should be a PaillierPublicKey')

    @property
    def public_key(self):
        return self._message['public_key']

    @property
    def __ciphertext(self):
        return self._message['ciphertext']

    @property
    def exponent(self):
        return self._message['exponent']

    @property
    def __is_obfuscated(self):
        return self._message['is_obfuscated']

    @public_key.setter
    def public_key(self, public_key):
        self._message['public_key'] = public_key

    @__ciphertext.setter
    def __ciphertext(self, ciphertext):
        self._message['ciphertext'] = ciphertext

    @exponent.setter
    def exponent(self, exponent):
        self._message['exponent'] = exponent

    @__is_obfuscated.setter
    def __is_obfuscated(self, is_obfuscated):
        self._message['is_obfuscated'] = is_obfuscated

    def __add__(self, other):
        """Add an int, float, `PaillierCiphertext` or `EncodedNumber`."""
        if isinstance(other, PaillierCiphertext):
            return self._add_encrypted(other)
        elif isinstance(other, EncodedNumber):
            return self._add_encoded(other)
        else:
            return self._add_scalar(other)

    def __radd__(self, other):
        """Called when Python evaluates `34 + <PaillierCiphertext>`
        Required for builtin `sum` to work.
        """
        return self.__add__(other)

    def __mul__(self, other):
        """Multiply by an int, float, or EncodedNumber."""
        if isinstance(other, PaillierCiphertext):
            raise NotImplementedError('Good luck with that...')

        if isinstance(other, EncodedNumber):
            encoding = other
        else:
            encoding = EncodedNumber.encode(self.public_key, other)
        product = self._raw_mul(encoding.encoding)
        exponent = self.exponent + encoding.exponent

        return PaillierCiphertext(self.public_key, product, exponent)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self + (other * -1)

    def __rsub__(self, other):
        return other + (self * -1)

    def __truediv__(self, scalar):
        return self.__mul__(1 / scalar)

    def __neg__(self):
        return self * -1

    def ciphertext(self, be_secure=True):
        """Return the ciphertext of the PaillierCiphertext.
        Choosing a random number is slow. Therefore, methods like
        :meth:`__add__` and :meth:`__mul__` take a shortcut and do not
        follow Paillier encryption fully - every encrypted sum or
        product should be multiplied by r **
        :attr:`~PaillierPublicKey.n` for random r < n (i.e., the result
        is obfuscated). Not obfuscating provides a big speed up in,
        e.g., an encrypted dot product: each of the product terms need
        not be obfuscated, since only the final sum is shared with
        others - only this final sum needs to be obfuscated.
        Not obfuscating is OK for internal use, where you are happy for
        your own computer to know the scalars you've been adding and
        multiplying to the original ciphertext. But this is *not* OK if
        you're going to be sharing the new ciphertext with anyone else.
        So, by default, this method returns an obfuscated ciphertext -
        obfuscating it if necessary. If instead you set `be_secure=False`
        then the ciphertext will be returned, regardless of whether it
        has already been obfuscated. We thought that this approach,
        while a little awkward, yields a safe default while preserving
        the option for high performance.
        Args:
          be_secure (bool): If any untrusted parties will see the
            returned ciphertext, then this should be True.
        Returns:
          an int, the ciphertext. If `be_secure=False` then it might be
            possible for attackers to deduce numbers involved in
            calculating this ciphertext.
        """
        if be_secure and not self.__is_obfuscated:
            self.obfuscate()

        return self.__ciphertext

    def decrease_exponent_to(self, new_exp):
        """Return an EncryptedNumber with same value but lower exponent.
        If we multiply the encoded value by :attr:`EncodedNumber.BASE` and
        decrement :attr:`exponent`, then the decoded value does not change.
        Thus we can almost arbitrarily ratchet down the exponent of an
        `EncryptedNumber` - we only run into trouble when the encoded
        integer overflows. There may not be a warning if this happens.
        When adding `EncryptedNumber` instances, their exponents must
        match.
        This method is also useful for hiding information about the
        precision of numbers - e.g. a protocol can fix the exponent of
        all transmitted `EncryptedNumber` instances to some lower bound(s).
        Args:
          new_exp (int): the desired exponent.
        Returns:
          PaillierCiphertext: Instance with the same plaintext and desired
            exponent.
        Raises:
          ValueError: You tried to increase the exponent.
        """
        if new_exp > self.exponent:
            raise ValueError('New exponent %i should be more negative than '
                             'old exponent %i' % (new_exp, self.exponent))
        multiplied = self * pow(EncodedNumber.BASE, self.exponent - new_exp)
        multiplied.exponent = new_exp
        return multiplied

    def obfuscate(self):
        """Disguise ciphertext by multiplying by r ** n with random r.
        This operation must be performed for every `PaillierCiphertext`
        that is sent to an untrusted party, otherwise eavesdroppers
        might deduce relationships between this and an antecedent
        `PaillierCiphertext`.
        For example::
            enc = public_key.encrypt(1337)
            send_to_nsa(enc)       # NSA can't decrypt (we hope!)
            product = enc * 3.14
            send_to_nsa(product)   # NSA can deduce 3.14 by bruteforce attack
            product2 = enc * 2.718
            product2.obfuscate()
            send_to_nsa(product)   # NSA can't deduce 2.718 by bruteforce attack
        """
        r = self.public_key.get_random_lt_n()
        r_pow_n = IntegerArithmetic.powmod(r, self.public_key.n, self.public_key.nsquare)
        self.__ciphertext = self.__ciphertext * r_pow_n % self.public_key.nsquare
        self.__is_obfuscated = True

    def _add_scalar(self, scalar):
        """Returns E(a + b), given self=E(a) and b.
        Args:
          scalar: an int or float b, to be added to `self`.
        Returns:
          PaillierCiphertext: E(a + b), calculated by encrypting b and
            taking the product of E(a) and E(b) modulo
            :attr:`~PaillierPublicKey.n` ** 2.
        Raises:
          ValueError: if scalar is out of range or precision.
        """
        encoded = EncodedNumber.encode(self.public_key, scalar,
                                       max_exponent=self.exponent)

        return self._add_encoded(encoded)

    def _add_encoded(self, encoded):
        """Returns E(a + b), given self=E(a) and b.
        Args:
          encoded (EncodedNumber): an :class:`EncodedNumber` to be added
            to `self`.
        Returns:
          PaillierCiphertext: E(a + b), calculated by encrypting b and
            taking the product of E(a) and E(b) modulo
            :attr:`~PaillierPublicKey.n` ** 2.
        Raises:
          ValueError: if scalar is out of range or precision.
        """
        if self.public_key != encoded.public_key:
            raise ValueError("Attempted to add numbers encoded against "
                             "different public keys!")

        # In order to add two numbers, their exponents must match.
        a, b = self, encoded
        if a.exponent > b.exponent:
            a = self.decrease_exponent_to(b.exponent)
        elif a.exponent < b.exponent:
            b = b.decrease_exponent_to(a.exponent)

        # Don't bother to salt/obfuscate in a basic operation, do it
        # just before leaving the computer.
        encrypted_scalar = a.public_key.raw_encrypt(b.encoding, 1)

        sum_ciphertext = a._raw_add(a.ciphertext(False), encrypted_scalar)
        return PaillierCiphertext(a.public_key, sum_ciphertext, a.exponent)

    def _add_encrypted(self, other):
        """Returns E(a + b) given E(a) and E(b).
        Args:
          other (PaillierCiphertext): an `EncryptedNumber` to add to self.
        Returns:
          PaillierCiphertext: E(a + b), calculated by taking the product
            of E(a) and E(b) modulo :attr:`~PaillierPublicKey.n` ** 2.
        Raises:
          ValueError: if numbers were encrypted against different keys.
        """
        if self.public_key != other.public_key:
            raise ValueError("Attempted to add numbers encrypted against "
                             "different public keys!")

        # In order to add two numbers, their exponents must match.
        a, b = self, other
        if a.exponent > b.exponent:
            a = self.decrease_exponent_to(b.exponent)
        elif a.exponent < b.exponent:
            b = b.decrease_exponent_to(a.exponent)

        sum_ciphertext = a._raw_add(a.ciphertext(False), b.ciphertext(False))
        return PaillierCiphertext(a.public_key, sum_ciphertext, a.exponent)

    def _raw_add(self, e_a, e_b):
        """Returns the integer E(a + b) given ints E(a) and E(b).
        N.B. this returns an int, not an `PaillierCiphertext`, and ignores
        :attr:`ciphertext`
        Args:
          e_a (int): E(a), first term
          e_b (int): E(b), second term
        Returns:
          int: E(a + b), calculated by taking the product of E(a) and
            E(b) modulo :attr:`~PaillierPublicKey.n` ** 2.
        """
        return e_a * e_b % self.public_key.nsquare

    def _raw_mul(self, plaintext):
        """Returns the integer E(a * plaintext), where E(a) = ciphertext
        Args:
          plaintext (int): number by which to multiply the
            `PaillierCiphertext`. *plaintext* is typically an encoding.
            0 <= *plaintext* < :attr:`~PaillierPublicKey.n`
        Returns:
          int: Encryption of the product of `self` and the scalar
            encoded in *plaintext*.
        Raises:
          TypeError: if *plaintext* is not an int.
          ValueError: if *plaintext* is not between 0 and
            :attr:`PaillierPublicKey.n`.
        """
        if not isinstance(plaintext, int):
            raise TypeError('Expected ciphertext to be int, not %s' %
                            type(plaintext))

        if plaintext < 0 or plaintext >= self.public_key.n:
            raise ValueError('Scalar out of bounds: %i' % plaintext)

        if self.public_key.n - self.public_key.max_int <= plaintext:
            # Very large plaintext, play a sneaky trick using inverses
            neg_c = IntegerArithmetic.invert(self.ciphertext(False), self.public_key.nsquare)
            neg_scalar = self.public_key.n - plaintext
            return IntegerArithmetic.powmod(neg_c, neg_scalar, self.public_key.nsquare)
        else:
            return IntegerArithmetic.powmod(self.ciphertext(False), plaintext, self.public_key.nsquare)