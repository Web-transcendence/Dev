import { fetch } from 'undici'

type FriendList = {
	acceptedIds: number[];
	pendingIds: number[];
	receivedIds: number[];
};

export const fetchAcceptedFriends = async (id: number) => {
	const friends = await fetch(`http://social:6500/list`, {
		method: 'GET',
		headers: {
			'Content-Type': 'application/json',
			'id': id.toString(),
		}
	})
	if (friends.ok) {
		const friendIds = await friends.json() as FriendList
		return friendIds.acceptedIds
	}


}

