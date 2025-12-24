using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections;
using System;
using Random = UnityEngine.Random;
using UnityEngine.InputSystem;


public class DoggyAgent : Agent
{
    [Header("Сервоприводы")]
    public ArticulationBody[] legs;

    [Header("Скорость работы сервоприводов")]
    public float servoSpeed;

    [Header("Тело")]
    public ArticulationBody body;
    private Vector3 defPos;
    private Quaternion defRot;
    public float strenghtMove;

    [Header("Куб (цель)")]
    public GameObject cube;

    [Header("Сенсоры")]
    public Unity.MLAgentsExamples.GroundContact[] groundContacts;


    [Header("Параметры обучения")]
    public int rewardWarmupSteps = 100000;
    public int cubeWarmupSteps = 1000000;

    public float distanceRewardScale = 0.5f;
    public float velocityRewardScale = 0.02f;
    public float aligmentRewardScale = 0.002f;
    public float contactPerLegReward = 0.005f;

    public float energyPenaltyScale = -0.0003f;
    public float timePenalty = -0.001f;

    public float successReward = 5.0f;
    public float successDistance = 0.5f;
    

    private float prevDistance;
    private int globalSteps;
    

    public override void Initialize()
    {
        defPos = body.transform.position;
        defRot = body.transform.rotation;
    }
    
    public void ResetDog()
    {
        body.TeleportRoot(defPos, defRot);
        body.velocity = Vector3.zero;
        body.angularVelocity = Vector3.zero;

        for (int i = 0; i < 12; i++)
        {
            MoveLeg(legs[i], 0);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        Debug.Log("Heuristic");
    }

    public override void OnEpisodeBegin()
    {
    	ResetDog();

        float radius;
	if (!Academy.Instance.IsCommunicatorOn)
	{
	    radius = 7.5f;
	}
	else
	{
	    float t = cubeWarmupSteps == 0
		? 1f
		: Mathf.Clamp01((float)globalSteps / cubeWarmupSteps);

	    float rawRadius = Mathf.Lerp(0f, 10f, t);
	    radius = Mathf.Clamp(rawRadius, 2.5f, 7.5f);
	}

	cube.transform.position = new Vector3(
	    Random.Range(-radius, radius),
	    0.21f,
	    Random.Range(-radius, radius)
	);
	
        prevDistance = Vector3.Distance(body.transform.position, cube.transform.position);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(body.transform.position);
        sensor.AddObservation(body.velocity);
        sensor.AddObservation(body.angularVelocity);
        sensor.AddObservation(body.transform.right);

        sensor.AddObservation(cube.transform.position);

        Vector3 relativePosition = cube.transform.position - body.transform.position;
        sensor.AddObservation(relativePosition);

        Vector3 toCube = (cube.transform.position - body.transform.position).normalized;
        float angleToCube = Vector3.SignedAngle(body.transform.right, toCube, Vector3.up);
        sensor.AddObservation(angleToCube);

        float distanceToCube = Vector3.Distance(body.transform.position, cube.transform.position);
        sensor.AddObservation(distanceToCube);
        foreach (var leg in legs)
        {
            sensor.AddObservation(leg.xDrive.target);
            sensor.AddObservation(leg.velocity);
            sensor.AddObservation(leg.angularVelocity);
        }

        foreach(var groundContact in groundContacts)
        {
            sensor.AddObservation(groundContact.touchingGround);
        }
    }

    public override void OnActionReceived(ActionBuffers vectorAction)
    {
        var actions = vectorAction.ContinuousActions;
        for (int i = 0; i < 12; i++)
        {
            float angle = Mathf.Lerp(legs[i].xDrive.lowerLimit, legs[i].xDrive.upperLimit, (actions[i] + 1) * 0.5f);
            MoveLeg(legs[i], angle);
        }

        globalSteps++;

        float warmup = rewardWarmupSteps == 0 ? 1f : Mathf.Clamp01((float)globalSteps / rewardWarmupSteps);
    
        Vector3 toCube = cube.transform.position - body.transform.position;
        Vector3 direction = toCube.normalized;
        float distance = toCube.magnitude;
        
        float delta = prevDistance - distance;
        prevDistance = distance;
        AddReward(warmup * distanceRewardScale * delta);

        float velocityProjection = Vector3.Dot(body.velocity, direction);
        AddReward(warmup * velocityRewardScale * Mathf.Clamp(velocityProjection, -1f, 1f));

        float alignment = Vector3.Dot(body.transform.right.normalized, direction);
	AddReward(warmup * aligmentRewardScale * alignment);

        int contacts = 0;
        foreach (var gc in groundContacts)
            if (gc.touchingGround) contacts++;

        AddReward(warmup * contactPerLegReward * contacts);

        float energy = 0f;
        for (int i = 0; i < 12; i++)
            energy += Mathf.Abs(actions[i]);
            
        AddReward(warmup * energyPenaltyScale * energy);

        AddReward(timePenalty);

        if (distance < successDistance)
        {
            AddReward(successReward);
            EndEpisode();
        }
    }

    public void FixedUpdate()
    {
        Debug.DrawRay(body.transform.position, body.transform.right, Color.white);
    }

    void MoveLeg(ArticulationBody leg, float targetAngle)
    {
        leg.GetComponent<Leg>().MoveLeg(targetAngle, servoSpeed);
    }
}
