import {FastifyInstance, FastifyReply, FastifyRequest} from "fastify";
import {fetchIdByNickName} from "./utils.js";
import {generateId, initSchema, inputHandler, inputSchema, Player, readySchema, resetInput,} from "./api.js";
import {getMatchHistory, MatchResult} from "./database.js"
import {soloMode} from "./solo.js";
import {generateRoom, joinRoom, leaveRoom} from "./netcode.js";

export default async function pongRoutes(fastify: FastifyInstance) {
    fastify.get('/ws', { websocket: true }, (socket, req) => {
        console.log("Client connected");
        const userId = generateId();
        const player = new Player(userId, socket);
        let init = false;
        let room = -1;
        let solo: boolean;
        socket.on("message", async (message) => {
            const msg = JSON.parse(message.toString());
            if (!init && msg.type === "socketInit") {
                const {data, success, error} = initSchema.safeParse(JSON.parse(message.toString()));
                if (!success || !data) {
                    console.error(error);
                    return ;
                }
                player.name = data.nick;
                player.paddle.name = data.nick;
                if (player.name === "AI")
                    player.frequency = 1000;
                try {
                    player.dbId = await fetchIdByNickName(data.nick);
                } catch (error) {
                    console.log(error);
                }
                if (data.room)
                    room = msg.room;
                init = true;
            } else if (msg.type === "input") {
                const {data, success, error} = inputSchema.safeParse(JSON.parse(message.toString()));
                if (!success || !data) {
                    console.error(error);
                    return ;
                }
                resetInput(player.input);
                inputHandler(data.key, data.state, player.input);
            } else if (init && msg.type === "ready") {
                const {data, success, error} = readySchema.safeParse(JSON.parse(message.toString()));
                if (!success || !data) {
                    console.error(error);
                    return ;
                }
                console.log(data.mode);
                if (data.mode === "remote")
                    joinRoom(player, room);
                else if (data.mode === "local") {
                    solo = true;
                    soloMode(player, solo);
                }
            }
        });
        socket.on("close", () => {
            leaveRoom(userId);
            solo = false;
            console.log("Client disconnected");
        });
    });

    fastify.post('/generateRoom', async (req: FastifyRequest, res: FastifyReply) => {
        const roomId = generateRoom();
        return (res.status(200).send({roomId: roomId}));
    })

    fastify.get('/getMatchHistory', async (req: FastifyRequest, res: FastifyReply) => {
        try {
            const id = Number(req.headers.id);
            const MatchResult: MatchResult[] = getMatchHistory(id);
            return (res.status(200).send(MatchResult));
        } catch (error) {
            console.log(error);
            return (res.status(500).send({error}));
        }
    })
}
