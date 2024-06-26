#pragma once
#include "forceline.h"

class tri3f {
public:
	unsigned int _ids[3];

	FORCEINLINE tri3f() {
		_ids[0] = _ids[1] = _ids[2] = -1;
	}

	FORCEINLINE tri3f(unsigned int id0, unsigned int id1, unsigned int id2) {
		set(id0, id1, id2);
	}

	FORCEINLINE void set(unsigned int id0, unsigned int id1, unsigned int id2) {
		_ids[0] = id0;
		_ids[1] = id1;
		_ids[2] = id2;
	}

	FORCEINLINE unsigned int id(int i) { return _ids[i]; }
	FORCEINLINE unsigned int id0() {return _ids[0];}
	FORCEINLINE unsigned int id1() {return _ids[1];}
	FORCEINLINE unsigned int id2() {return _ids[2];}
	FORCEINLINE void reverse() {std::swap(_ids[0], _ids[2]);}
};

//use for unremove duplications
class stri3f {
public:
	unsigned int _ids[3];
	unsigned int _sids[3];

	FORCEINLINE stri3f() {
		set(-1, -1, -1);
	}

	FORCEINLINE stri3f(unsigned int id0, unsigned int id1, unsigned int id2) {
		set(id0, id1, id2);
	}

	FORCEINLINE void set(unsigned int id0, unsigned int id1, unsigned int id2) {
		_sids[0] = _ids[0] = id0;
		_sids[1] = _ids[1] = id1;
		_sids[2] = _ids[2] = id2;

		std::sort(std::begin(_sids), std::end(_sids));
	}

	FORCEINLINE tri3f getTri() const {
		return tri3f(_ids[0], _ids[1], _ids[2]);
	}

	FORCEINLINE unsigned int id(int i) { return _ids[i]; }
	FORCEINLINE unsigned int id0() { return _ids[0]; }
	FORCEINLINE unsigned int id1() { return _ids[1]; }
	FORCEINLINE unsigned int id2() { return _ids[2]; }
	FORCEINLINE void reverse() { std::swap(_ids[0], _ids[2]); }

	bool operator < (const stri3f& other) const {
		if (_sids[0] == other._sids[0]) {
			if (_sids[1] == other._sids[1])
				return _sids[2] < other._sids[2];
			else
				return _sids[1] < other._sids[1];
		}
		else
			return _sids[0] < other._sids[0];
	}

	bool operator == (const stri3f& other) const {
		return (_sids[0] == other._sids[0] && _sids[1] == other._sids[1] && _sids[2] == other._sids[2]);
	}

};

