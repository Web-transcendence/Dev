
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
            const parse = JSON.parse(value?.replace('data: ', ''));
            if (parse.event in mapEvent) {
                console.log(parse)
                mapEvent[parse.event](parse.data);
            }
        }
    } catch (err) {
        console.error(err);
    }
}


const notifyNewFriend = ({id}: { id: number }) => {
    console.log(`this id ${id} have to be add in the friendlist`)

}

const notifyFriendInvitation = ({id1, id2}: {id1: number, id2: number}) => {
    console.log(`this friend have to be add in the invitation list`)
}

const notifyFriendRemoved = ({id}: { id: number }) => {
    console.log(`this friend have to be supressed from the friendlist`)
}

const notifyDisconnection = ({id}: { id: number }) => {
    console.log(`this friend have to be marked as unconnected`)
}

const notifyConnection = ({id}: { id: number }) => {
    console.log(`this friend have to be marked as connected`)
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