import * as Schema from "./schema.js";
import sanitizeHtml from "sanitize-html";
import {User} from "./User.js";
import { FastifyReply, FastifyRequest, FastifyInstance } from "fastify";
import {connectedUsers} from "./api.js"
import {EventMessage} from "fastify-sse-v2";



export default async function userRoutes(app: FastifyInstance) {

    app.post('/sign-up', async (req, res) => {
        try {
            const zod_result = Schema.signUpSchema.safeParse(req.body);
            if (!zod_result.success)
                return res.status(400).send({json: zod_result.error.format()});

            let {nickName, email, password} = {nickName: sanitizeHtml(zod_result.data.nickName), email: sanitizeHtml(zod_result.data.email), password: sanitizeHtml(zod_result.data.password)};
            if (!nickName || !email || !password)
                return res.status(454).send({error: "All information are required !"});

            const addRes = await User.addClient(nickName, email, password);
            if (!addRes.success) {
                return res.status(409).send({error: `${addRes.result} already exists`});
            }
            return res.status(201).send({token: addRes.result, redirect: "post/login"});
        }
        catch(err) {
            return res.status(500).send({error: "Server error: ", err});
        }
    });

    app.get('/getProfile', async (req, res) => {

        try {
            const zod_result = Schema.profileSchema.safeParse(req.headers);
            if (!zod_result.success)
                return res.status(400).send({json: zod_result.error.format()});
            const id = sanitizeHtml(zod_result.data.id);
            if (!id)
                return res.status(454).send({error: "All information are required !"});

            const user = new User(id);

            user.sendNotification();
            const profileData = user.getProfile();

            return res.status(200).send(profileData);
        } catch (err) {
            return res.status(500).send({error: "Server error: ", err});
        }


    })

    app.post('/sign-in', async (req: FastifyRequest, res: FastifyReply) => {
        try {
            console.log("sign-in");
            const zod_result = Schema.signInSchema.safeParse(req.body);
            if (!zod_result.success)
                return res.status(400).send({json: zod_result.error.format()});
            let {nickName, password} = {
                nickName: sanitizeHtml(zod_result.data.nickName),
                password: sanitizeHtml(zod_result.data.password)
            };
            if (!nickName || !password)
                return res.status(454).send({error: "All information are required !"});

            const connection = await User.login(nickName, password);
            if (connection.code !== 201)
                return res.status(connection.code).send({error: connection.result});
            return res.status(200).send({token: connection.result, redirect: "post/login"});
        }
        catch (err) {
            res.status(500).send({error: "Server error: ", err});
        }
    });


    app.get('/sse', async function (req, res) {
        const userId = req.headers.id as string;
        if (!userId)
            return res.status(500).send({error: "Server error: Id not found"});
        connectedUsers.set(userId, res);
        const message: EventMessage = { event: "inititation", data: "Some message" }
        res.sse({data: JSON.stringify(message)});
    });

}