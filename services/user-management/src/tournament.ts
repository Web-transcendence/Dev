import {tournamentSessions} from "./api.js"
import {User} from "./User.js"

export class tournament {

    creatorId: number
    private participantId: number[]
    private status: 'waiting' | 'started'
    private result

    constructor(creatorId: number) {
        this.creatorId = creatorId
        this.participantId = [this.creatorId]
        this.status = 'waiting'
        this.result = {}
    }

    hasParticipant(userId: number): boolean {
        return this.participantId.includes(userId)
    }

    addParticipant(participantId: number) {
        if (this.status === 'started')
            throw new ConflictError(`this tournament has already started`)

        for (const [id, tournament] of tournamentSessions)
            if (tournament.hasParticipant(participantId))
                throw new ConflictError(`this user has already another tournament`)

        this.participantId.push(participantId)
    }

    getData(): {creatorId: number, creatorNickName: string, participantCount: number, status: string} {
        const user = new User(this.creatorId)
        const data = {
            creatorId: this.creatorId,
            creatorNickName: user.getProfile().nickName,
            participantCount: this.participantId.length,
            status : this.status,
        }
        return data
    }

    quit(id: number): void {
        this.participantId.filter(participantId => participantId !== id)
    }

    async bracketHandler(bracket: number[]): Promise<number> {
        if (bracket.length === 2) {
            //await startMatch(bracket[0], bracket[1])
            console.log(`Match entre ${bracket[0]} et ${bracket[1]}`)
            await new Promise(res => setTimeout(res, 1000))
            console.log(`Match terminé: ${bracket[0]} vs ${bracket[1]}`)
            return bracket[Math.floor(Math.random() > 0.5 ? 0 : 1)]
        }
        else {
            const winner = await Promise.all([
                this.bracketHandler(bracket.slice(0, bracket.length / 2)),
                this.bracketHandler(bracket.slice(bracket.length / 2))
            ])

            //await startMatch(bracket[0], bracket[1])
            console.log(`Match entre ${winner[0]} et ${winner[1]}`)
            await new Promise(res => setTimeout(res, 1000))
            console.log(`Match terminé: ${winner[0]} vs ${winner[1]}`)
            return bracket[Math.floor(Math.random() > 0.5 ? 0 : 1)]
        }
    }

    async launch(): Promise<string> {
        if (this.status === 'started')
            throw new ConflictError(`this tournament has already started`)
        if (![4, 8, 16, 32].includes(this.participantId.length))
            throw new ConflictError(`participant have to be 4, 8, 16 or 32`)

        const arr = [...this.participantId]
        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [arr[i], arr[j]] = [arr[j], arr[i]]
        }

        const winner = await this.bracketHandler(arr)
        console.log(`${winner} is the winner`)

        return winner.toString()
    }


}


