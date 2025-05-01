import { fetch } from 'undici';


export const fetchAcceptedFriends = async (id: number) => {
    const friends = await fetch(`http://social:6000/test`)
    if (friends.ok) {
        const friendIds = await friends.json()
        return friendIds.acceptedIds
    }


}

