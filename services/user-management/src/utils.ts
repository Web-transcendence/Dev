import { fetch } from 'undici';


export const fetchAcceptedFriends = async (id: number) => {
    const friends = await fetch(`http://social:6500/list`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'id': id,
        }
    })
    if (friends.ok) {
        const friendIds = await friends.json()
        return friendIds.acceptedIds
    }


}

