#pragma once

class id_pair {
	unsigned int _id[2];

public:
	id_pair() {
		_id[0] = _id[1] = -1;
	}

	id_pair(unsigned int id1, unsigned int id2, bool sort)
	{
		_id[0] = id1;
		_id[1] = id2;

		if (sort && id1 > id2) {
			_id[0] = id2;
			_id[1] = id1;
		}
	}

	void get(unsigned int& id1, unsigned int& id2)
	{
		id1 = _id[0];
		id2 = _id[1];
	}

	bool operator < (const id_pair& other) const {
		if (_id[0] == other._id[0])
			return _id[1] < other._id[1];
		else
			return _id[0] < other._id[0];
	}

	bool operator == (const id_pair& other) const {
		return (_id[0] == other._id[0] && _id[1] == other._id[1]);
	}

};
