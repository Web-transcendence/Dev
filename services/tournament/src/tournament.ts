import {INTERNAL_PASSWORD, tournamentSessions} from "./api.js"
import {ConflictError, ServerError} from "./error.js";
import {fetchNotifyUser} from "./utils.js";
import {fetch} from "undici"


async function fetchMatch(id1: number, id2: number) {
    const response = await fetch(`http://match-server:4443/tournamentGame`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'authorization': `${INTERNAL_PASSWORD}`
        },
        body: JSON.stringify({id1: id1, id2: id2})
    })
    if (response.ok) {
        const winner = await response.json()
        return winner.id
    }
    throw new Error(`CREATE A REAL ERROR MESSAGE`)
}

export class tournament {

    private participantId: number[]
    private status: 'waiting' | 'started'
    maxPlayer: number
    private actualParticipant: number[]
    private alonePlayerId: number

    constructor(maxPlayer: number) {
        if ([4, 8, 16, 32].includes(maxPlayer))
            this.maxPlayer = maxPlayer
        else
            throw new ServerError(`cannot create a tournament with ${maxPlayer} player`, 500)
        this.participantId = []
        this.status = 'waiting'
        this.actualParticipant = []
    }

    hasParticipant(userId: number): boolean {
        return this.participantId.includes(userId)
    }
    hasPlayer(userId: number): boolean {
        return this.actualParticipant.includes(userId)
    }

    async addParticipant(participantId: number) {
        if (this.status === 'started')
            throw new ConflictError(`this tournament has already started`, `This tournament is already started`)
        if (this.participantId.length >= this.maxPlayer)
            throw new ConflictError(`maxPlayer has already is reached`, `This tournament is full`)

        for (const [id, tournament] of tournamentSessions)
            if (tournament.hasParticipant(participantId))
                throw new ConflictError(`this user has already another tournament`, `you cannot participate to multiple tournament`)

        await fetchNotifyUser(this.participantId, 'joinTournament', {id: participantId, maxPlayer: this.maxPlayer})
        this.participantId.push(participantId)
    }

    getData(): {participants: number[], maxPlayer: number, status: string} {
        return {
            participants: this.participantId,
            maxPlayer: this.maxPlayer,
            status: this.status,
        }
    }

    async quit(id: number){
        if (this.status === 'started') {
            return ;
        }
        else {
            this.participantId = this.participantId.filter(participantId => participantId !== id)
            await fetchNotifyUser(this.participantId, 'quitTournament', {id: id, maxPlayer: this.maxPlayer})
        }
    }

    async bracketHandler() {
        if (this.alonePlayerId) {
            this.actualParticipant.push(this.alonePlayerId)
            this.alonePlayerId = 0
        }

        let i = 0

        for (; i < this.actualParticipant.length - 1; i += 2) {
           await fetchMatch(this.actualParticipant[i], this.actualParticipant[i + 1])
        }

        if (i !== this.actualParticipant.length) {
            this.alonePlayerId = this.actualParticipant[i]
            this.actualParticipant.splice(i, 1)
        }

    }


    async launch(): Promise<string> {
        if (this.status === 'started')
            throw new ConflictError(`this tournament has already started`, `cannot launch this tournament`)
        if (this.participantId.length !== this.maxPlayer)
            throw new ConflictError(`participant have to be 4, 8, 16 or 32`, `cannot launch this tournament without ${this.maxPlayer} participant`)

        for (let i = this.participantId.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [this.participantId[i], this.participantId[j]] = [this.participantId[j], this.participantId[i]]
        }

        this.actualParticipant = this.participantId
        console.log(`launch tournament`)
        await this.bracketHandler()
    }

    async bracketWon(id: number) {
        console.log(`${id} win a bracket`)
        let loser: number
        let winner: number
        let winnerEvent: string = `winBracket`
        for (let i = 0 ; i < this.actualParticipant.length; i++) {
            if (id === this.actualParticipant[i] && i % 2 === 0) {
                loser = this.actualParticipant[i + 1]
                this.actualParticipant[i + 1] = 0

                winner = this.actualParticipant[i]
                break
            }
            else if (id === this.actualParticipant[i]) {
                loser = this.actualParticipant[i - 1]
                this.actualParticipant[i - 1] = 0

                winner = this.actualParticipant[i]
                break
            }
        }
        if (!loser || !winner)
            throw new ServerError(`internal server error, it doesn't happen`, 500)

        if (this.actualParticipant.length === 2 && this.alonePlayerId == 0)
            winnerEvent = `winTournament`

        await fetchNotifyUser([loser], 'loseTournament', {})
        await fetchNotifyUser([winner], winnerEvent, {})

        if (winnerEvent === 'winTournament') {
            console.log(`tournament finished`)
            this.cleanTournament()
        }
        else if (this.actualParticipant.filter(id => id !== 0).length * 2 === this.actualParticipant.length) {
            console.log(`bracket ended, start the next`)
            this.actualParticipant = this.actualParticipant.filter(id => id !== 0)
            await this.bracketHandler()
        }
    }

    cleanTournament() {
        this.participantId = []
        this.status = 'waiting'
        this.actualParticipant = []
        this.alonePlayerId = 0
    }

    // sessionData() {
    //     const bracket: [{ id1: number, id2: number}] = []
    //
    //     let i = 1
    //
    //     for (; i < this.actualParticipant.length - 1; i += 2) {
    //         bracket.push({id1: this.actualParticipant[i], id2: this.actualParticipant[i + 1]})
    //     }
    //
    //     if (i !== this.actualParticipant.length) {
    //         bracket.push({id1: this.actualParticipant[i], id2: 0})
    //     }
    //
    // }
}


