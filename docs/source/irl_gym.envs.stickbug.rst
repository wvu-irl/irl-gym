.. _stickbug-landing:

Stickbug
========

The stickbug environment is a 3D kinematic simulation of a stickbug robot.
It lets you both move the robot base and arms with options for position and velocity control.
It additionally provides support for custom observation and pollination models.
For more information, see the :ref:`robot body <stickbug-robot>`, :ref:`orchard <orchard>`, :ref:`observation <stickbug-observation>`, and :ref:`pollination <stickbug-pollination>` sections.

Be aware, to run the Stickbug environment, you will need to supply a set of parameters for each of the above. 
For a sample please check the Github file `sb_params.json <https://github.com/wvu-irl/irl-gym/blob/main/irl_gym/test/sb_params.json>`_

Additionally be mindful that the observations will pass back whether a flower was pollinated. In the future this may be made partially observable.

Contents
---------

.. toctree::
   :maxdepth: 2

   irl_gym.envs.stickbug.body
   irl_gym.envs.stickbug.observation
   irl_gym.envs.stickbug.pollination
   irl_gym.envs.orchard

StickbugEnv
-----------

.. automodule:: irl_gym.envs.stickbug
   :members:
   :undoc-members:
   :show-inheritance: