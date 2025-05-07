import {addFriend, fetchUserInformation, removeFriend} from "./user.js";


const parseSSEMessage  = (raw: string): {event: string, stringData: string} => {
    const result: Record<string, string> = {}
    const lines = raw.split('\n')

    for (const line of lines) {
        const [key, ...rest] = line.split(':')
        result[key.trim()] = rest.join(':').trim()
    }
    return {event: result.event, stringData: result.data}
}

export async function sseConnection() {
    try {
        const token = localStorage.getItem('token')
        const res = await fetch(`/user-management/sse`, {
            method: 'GET',
            headers: {
                'Content-Type': 'text/event-stream',
                'Authorization': `Bearer ${token}`
            }
        })
        if (res.status === 100)
            return;
        if (!res.ok) {
            const error = res.json()
            console.log(error)
            //notifyhandling
            return ;
        }
        console.log('sse connection')
        const reader = res.body?.pipeThrough(new TextDecoderStream()).getReader() ?? null;
        while (reader) {
            const {value, done} = await reader.read();
            if (done) break;
            if (value.startsWith('retry: ')) continue;
            const parse = parseSSEMessage(value)
            if (parse.event in mapEvent)
                mapEvent[parse.event](JSON.parse(parse.stringData));
        }
    } catch (err) {
        console.error(err);
    }
}

export const CreateFriendLi = async (id: number, key: string, tmpName: string)=> {
    console.log(`this id ${id} have to be add in the friendlist`)
    document.getElementById(`friendId-${id}`)?.remove();
    const list = document.getElementById(key);
    const [userData] = await fetchUserInformation([id])
    const template = document.getElementById(tmpName) as HTMLTemplateElement | null;

    if (list && template) {
        const clone = template.content.cloneNode(true) as HTMLElement;
        const item = clone.querySelector("li");
        if (item) item.id = `friendId-${userData.id}`;
        const img = clone.querySelector("img");
        if (img && userData.avatar) img.src = userData.avatar;
        const name = clone.querySelector(".name");
        if (name) name.textContent = userData.nickName;
        console.log('logOnline :', userData.online);
        if (userData.online && key === "acceptedList")
            clone.querySelector(".online")?.classList.remove('hidden');
        if (key === "receivedList") {
            clone.querySelector(".accept-btn")?.addEventListener("click", () => addFriend(userData.nickName));
            clone.querySelector(".decline-btn")?.addEventListener("click", () => removeFriend(userData.nickName));
        }
        list.appendChild(clone);
    }
}


const notifyNewFriend = async ({id}: { id: number }) => {
    console.log(`this id ${id} have to be add in the friendlist`)
    await CreateFriendLi(id, "acceptedList", "acceptedTemplate")
}

const notifyFriendInvitation = async ({id}: {id: number}) => {
    console.log(`this friend have to be add in the invitation list`)
    await CreateFriendLi(id, "receivedList", "receivedTemplate")
}

const notifyFriendRemoved = ({id}: { id: number }) => {
    console.log(`this friend have to be supressed from the friendlist`)
    const list = document.getElementById(`friendId-${id}`);
    if (list) list.remove();
}

const notifyDisconnection = ({id}: { id: number }) => {
    console.log(`this friend have to be marked as unconnected`)
    const list = document.getElementById(`friendId-${id}`);
    if (list) list.querySelector(".online")?.classList.add('hidden');
}

const notifyConnection = ({id}: { id: number }) => {
    console.log(`this friend have to be marked as connected`)
    const list = document.getElementById(`friendId-${id}`);
    if (list) list.querySelector(".online")?.classList.remove('hidden');
}

const notifyJoinTournament = ({id}: { id: number }) => {
    console.log(`the user with the id ${id} joined my tournament`)

}

const notifyQuitTournament = ({id}: { id: number }) => {
    console.log(`the user with the id ${id} Quit my tournament`)
}


const mapEvent : {[key: string] : (data: any) => void} = {
    "joinTournament" : notifyJoinTournament,
    "quitTournament" : notifyQuitTournament,
    "newFriend" : notifyNewFriend,
    "friendInvitation" : notifyFriendInvitation,
    "friendRemoved": notifyFriendRemoved,
    "connection" : notifyConnection,
    "disconnection" : notifyDisconnection,

}